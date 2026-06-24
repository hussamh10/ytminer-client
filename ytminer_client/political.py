"""Political-transcripts worker mode (`--political-videos`).

Instead of downloading videos, this leases video ids from a *transcript coordinator*
(the political-videos-server / political-transcript-swarm), fetches each video's
English captions locally with yt-dlp + a local bgutil PO-token provider, and posts
the transcripts back. Many workers on different networks sidestep YouTube's
per-subnet caption rate limit.

Requirements for this mode:
  * a bgutil PO-token provider reachable at --provider-url (default
    http://127.0.0.1:4416), and the `bgutil-ytdlp-pot-provider` yt-dlp plugin.
    See https://github.com/Brainicism/bgutil-ytdlp-pot-provider (run the HTTP server).
  * the --server pointed at the transcript coordinator (e.g. http://host:8771).

Coordinator API used: GET /work?n&worker_id -> {"ids":[...]}, POST /results
{worker_id, results:[{id,status,lang,kind,chars,text,err}]}.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import click
import httpx
import yt_dlp

EN_RE = re.compile(r"^en")
CANARY_ID = "dQw4w9WgXcQ"  # Rick Astley — definitely has English captions
DEFAULT_LANGS = "en-orig,en,en-US,en-GB"  # real English only (NOT en.* = auto-translations)
GUARD_WINDOW = 200
GUARD_MIN_OK = 0.30
COOLDOWN_STEPS = [30 * 60, 60 * 60, 2 * 3600]  # 30m, 1h, 2h (then 2h)


def _lang_rank(l: str) -> int:
    order = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-orig", "en-en"]
    return order.index(l) if l in order else len(order)


def _parse_json3(path: str) -> str | None:
    try:
        d = json.loads(open(path, encoding="utf-8").read())
    except Exception:
        return None
    text = "".join(
        seg.get("utf8", "")
        for ev in d.get("events", []) or []
        for seg in (ev.get("segs", []) or [])
    )
    return re.sub(r"\s+", " ", text).strip()


class TranscriptFetcher:
    """One yt-dlp instance + temp dir per worker thread (reused across ids)."""

    def __init__(self, langs: list[str], provider_url: str, sock_timeout: int = 30):
        self.langs = langs
        self.provider_url = provider_url
        self.sock_timeout = sock_timeout
        self._tl = threading.local()

    def _ydl(self):
        if getattr(self._tl, "ydl", None) is None:
            tmp = tempfile.mkdtemp(prefix="ytminer_pol_")
            self._tl.tmp = tmp
            extr = {"youtube": {"player_client": ["default"]}}
            if self.provider_url:
                extr["youtubepot-bgutilhttp"] = {"base_url": [self.provider_url]}
            self._tl.ydl = yt_dlp.YoutubeDL({
                "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
                "subtitleslangs": self.langs, "subtitlesformat": "json3",
                "ignore_no_formats_error": True, "quiet": True, "no_warnings": True,
                "noprogress": True, "socket_timeout": self.sock_timeout,
                "retries": 3, "extractor_retries": 2, "fragment_retries": 3,
                "extractor_args": extr,
                "cachedir": os.path.join(tempfile.gettempdir(), "ytminer_ytdlp_cache"),
                "outtmpl": {"default": os.path.join(tmp, "%(id)s.%(ext)s")},
            })
        return self._tl.ydl, self._tl.tmp

    def handle(self, vid: str) -> dict:
        ydl, tmp = self._ydl()
        for f in glob.glob(os.path.join(tmp, glob.escape(vid) + "*")):
            try: os.remove(f)
            except OSError: pass
        info, err = None, None
        try:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        except Exception as e:
            err = str(e)[:150]  # a late track's 429 can raise after good en files were written

        def file_lang(p):
            m = re.search(re.escape(vid) + r"\.([\w-]+)\.json3$", os.path.basename(p))
            return m.group(1) if m else ""
        files = [f for f in glob.glob(os.path.join(tmp, glob.escape(vid) + "*.json3"))
                 if EN_RE.match(file_lang(f))]
        if files:
            files.sort(key=lambda p: _lang_rank(file_lang(p)))
            lang = file_lang(files[0])
            text = _parse_json3(files[0])
            for f in glob.glob(os.path.join(tmp, glob.escape(vid) + "*")):
                try: os.remove(f)
                except OSError: pass
            if text:
                manual = lang in ((info or {}).get("subtitles") or {})
                return {"id": vid, "status": "ok", "lang": lang,
                        "kind": "manual" if manual else "auto", "chars": len(text), "text": text}
            return {"id": vid, "status": "error", "err": "empty json3"}
        if err is not None:
            return {"id": vid, "status": "error", "err": err}
        if info is None:
            return {"id": vid, "status": "error", "err": "extract_info None"}
        subs = set(info.get("subtitles") or {}) | set(info.get("automatic_captions") or {})
        if any(l.startswith("en") for l in subs):
            return {"id": vid, "status": "error", "err": "en listed but not downloaded"}
        if subs:
            return {"id": vid, "status": "no_target_lang"}
        return {"id": vid, "status": "no_captions"}


def run_political(server_url: str, worker_name: str, provider_url: str,
                  delay: float, jitter: float, batch: int, concurrency: int,
                  langs: str = DEFAULT_LANGS, no_canary: bool = False,
                  idle_sleep: int = 30) -> int:
    """Lease ids -> fetch transcripts -> post results. Returns an exit code."""
    server_url = server_url.rstrip("/")
    fetcher = TranscriptFetcher([s.strip() for s in langs.split(",")], provider_url)
    http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    try:
        http.get(f"{server_url}/healthz", timeout=10).raise_for_status()
    except Exception as e:
        click.echo(f"Cannot reach transcript coordinator at {server_url}: {e}", err=True)
        return 1

    if not no_canary:
        c = fetcher.handle(CANARY_ID)
        if c.get("status") != "ok" or not c.get("chars"):
            click.echo(
                f"CANARY FAILED ({c.get('status')}): captions are not downloading from here.\n"
                f"  - Is a bgutil PO-token provider running at {provider_url}?\n"
                f"  - Is `bgutil-ytdlp-pot-provider` installed and yt-dlp current?\n"
                f"  - Or is this IP currently rate-limited by YouTube?", err=True)
            return 2
        click.echo(f"Canary OK ({c['chars']} chars) — provider + IP healthy.")

    click.echo(f"Political-transcripts mode | worker={worker_name}")
    click.echo(f"Coordinator: {server_url} | provider: {provider_url}")
    click.echo(f"delay={delay}±{jitter}s | concurrency={concurrency} | langs={langs}\n")

    window: list[int] = []
    session_ok = session_total = 0

    while True:
        try:
            ids = http.get(f"{server_url}/work",
                           params={"n": batch, "worker_id": worker_name}).json().get("ids", [])
        except Exception as e:
            click.echo(f"  coordinator unreachable ({e}); retry in {idle_sleep}s")
            time.sleep(idle_sleep)
            continue
        if not ids:
            click.echo(f"  no work available; sleeping {idle_sleep}s")
            time.sleep(idle_sleep)
            continue

        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                results = list(ex.map(fetcher.handle, ids))
        else:
            results = []
            for i, vid in enumerate(ids):
                if i > 0 and (delay or jitter):
                    time.sleep(delay + random.uniform(0, jitter))
                results.append(fetcher.handle(vid))

        nok = sum(1 for r in results if r["status"] == "ok")
        session_ok += nok
        session_total += len(results)
        window.extend(1 if r["status"] == "ok" else 0 for r in results)
        if len(window) > GUARD_WINDOW:
            window = window[-GUARD_WINDOW:]

        for attempt in range(5):
            try:
                http.post(f"{server_url}/results",
                          json={"worker_id": worker_name, "results": results}).raise_for_status()
                break
            except Exception as e:
                click.echo(f"  submit failed ({e}); retry {attempt+1}/5")
                time.sleep(5 * (attempt + 1))

        click.echo(f"  batch {len(results)}: {nok} ok | session {session_ok}/{session_total} ok")

        # soft-block guard: if ok-rate over the window collapses, this IP is being
        # throttled (degraded responses look like 'no_captions' and would pollute the
        # corpus). Cool down instead of fetching, until a canary recovers.
        if len(window) >= GUARD_WINDOW and sum(window) / len(window) < GUARD_MIN_OK:
            step = 0
            while True:
                w = COOLDOWN_STEPS[min(step, len(COOLDOWN_STEPS) - 1)]
                click.echo(f"  SOFT-BLOCK (ok-rate {sum(window)/len(window):.0%} over {GUARD_WINDOW}) "
                           f"— cooling down {w//60}min to avoid polluting...")
                time.sleep(w)
                t = fetcher.handle(CANARY_ID)
                if t.get("status") == "ok" and t.get("chars"):
                    click.echo("  block lifted — resuming.")
                    window = []
                    break
                step += 1
