"""yt-dlp wrapper with cookie fallback and error classification."""

from __future__ import annotations

import asyncio
import logging
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ytminer-client.downloader")


# ─── Error Classification ──────────────────────────────────────


def classify_error(stderr: str) -> str:
    """Classify yt-dlp error output into a category."""
    msg = stderr.lower()
    if any(x in msg for x in ["sign in to confirm", "bot", "429", "too many"]):
        return "bot_blocked"
    if any(x in msg for x in [
        "unavailable", "private", "removed", "deleted",
        "copyright", "geo restricted", "members only",
        "age restricted", "terminated", "not found",
        "has been removed", "is not available",
    ]):
        return "permanent"
    if any(x in msg for x in ["timeout", "timed out", "connection", "network"]):
        return "network"
    return "unknown"


# ─── Rate Limiter ──────────────────────────────────────────────


@dataclass
class RateLimiter:
    """Simple rate limiter with jitter and adaptive backoff."""
    base_delay: float = 5.0
    jitter: float = 3.0
    consecutive_errors: int = 0
    max_backoff: float = 300.0  # 5 min cap

    async def wait(self):
        """Wait the appropriate amount of time before next request."""
        delay = self.base_delay + random.uniform(0, self.jitter)
        if self.consecutive_errors > 0:
            backoff = min(2 ** self.consecutive_errors, self.max_backoff)
            delay += backoff
            logger.debug(f"Backoff: {backoff:.0f}s (consecutive errors: {self.consecutive_errors})")
        await asyncio.sleep(delay)

    def report_success(self):
        self.consecutive_errors = 0

    def report_error(self, category: str):
        if category == "bot_blocked":
            self.consecutive_errors += 1
        elif category == "network":
            self.consecutive_errors += 1
        # permanent errors don't affect backoff


# ─── Cookie Manager ────────────────────────────────────────────


AUTO_BROWSERS = ["chrome", "firefox", "brave", "edge", "safari", "chromium"]


@dataclass
class CookieManager:
    """Manages cookie escalation strategy.

    Starts with no cookies, escalates on bot detection:
    none → explicit cookies file → cookies-from-browser (auto-detect)
    """
    cookies_file: str | None = None
    cookies_from_browser: str | None = None
    _mode: str = "none"
    _escalated: bool = False
    _auto_browser: str | None = None

    def get_args(self) -> list[str]:
        """Return yt-dlp cookie arguments for current mode."""
        if self._mode == "cookies_file" and self.cookies_file:
            return ["--cookies", self.cookies_file]
        elif self._mode == "cookies_from_browser":
            browser = self.cookies_from_browser or self._auto_browser
            if browser:
                return ["--cookies-from-browser", browser]
        return []

    def escalate(self) -> bool:
        """Try next cookie strategy. Returns True if escalation happened."""
        if self._mode == "none":
            if self.cookies_file:
                self._mode = "cookies_file"
                logger.info(f"Escalating to cookies file: {self.cookies_file}")
                return True
            elif self.cookies_from_browser:
                self._mode = "cookies_from_browser"
                logger.info(f"Escalating to cookies from browser: {self.cookies_from_browser}")
                return True
            else:
                # Auto-detect browser cookies
                browser = self._detect_browser()
                if browser:
                    self._auto_browser = browser
                    self._mode = "cookies_from_browser"
                    logger.info(f"Escalating to auto-detected browser cookies: {browser}")
                    return True
        elif self._mode == "cookies_file":
            if self.cookies_from_browser:
                self._mode = "cookies_from_browser"
                logger.info(f"Escalating to cookies from browser: {self.cookies_from_browser}")
                return True
            else:
                browser = self._detect_browser()
                if browser:
                    self._auto_browser = browser
                    self._mode = "cookies_from_browser"
                    logger.info(f"Escalating to auto-detected browser cookies: {browser}")
                    return True

        if not self._escalated:
            self._escalated = True
            logger.warning("Bot detection and no cookie strategies worked.")
        return False

    def _detect_browser(self) -> str | None:
        """Try each browser to see if yt-dlp can extract cookies from it."""
        yt_dlp_bin = str(Path(sys.executable).parent / "yt-dlp")
        if not Path(yt_dlp_bin).exists():
            yt_dlp_bin = shutil.which("yt-dlp") or "yt-dlp"

        for browser in AUTO_BROWSERS:
            try:
                proc = subprocess.run(
                    [yt_dlp_bin, "--cookies-from-browser", browser,
                     "--skip-download", "--no-warnings", "-q",
                     "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
                    capture_output=True, timeout=15,
                )
                # If it didn't error about the browser, it worked
                stderr = proc.stderr.decode(errors="replace").lower()
                if "no supported browser" not in stderr and "could not find" not in stderr and "error" not in stderr:
                    logger.info(f"Detected browser with cookies: {browser}")
                    return browser
            except Exception:
                continue
        return None

    @property
    def mode(self) -> str:
        return self._mode


# ─── Downloader ─────────────────────────────────────────────────


@dataclass
class DownloadResult:
    video_id: str
    status: str  # "ok", "failed", "skipped"
    error_category: str | None = None
    file_size: int = 0
    elapsed: float = 0.0


async def download_video(
    video_id: str,
    output_dir: Path,
    cookie_manager: CookieManager,
    timeout: int = 600,
) -> DownloadResult:
    """Download a video + metadata with yt-dlp.

    Returns DownloadResult with status and error info.
    """
    video_path = output_dir / f"{video_id}.mp4"
    info_path = output_dir / f"{video_id}.info.json"

    # Skip if both files already exist
    if video_path.exists() and info_path.exists():
        return DownloadResult(video_id=video_id, status="skipped")

    t0 = time.monotonic()

    # Find yt-dlp in the same venv as this Python process
    yt_dlp_bin = str(Path(sys.executable).parent / "yt-dlp")
    if not Path(yt_dlp_bin).exists():
        yt_dlp_bin = shutil.which("yt-dlp") or "yt-dlp"

    cmd = [
        yt_dlp_bin,
        "-f", "best[ext=mp4]/best",
        "-o", str(video_path),
        "--write-info-json",
        "--no-overwrites",
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "2",
        "--no-warnings",
    ]
    cmd.extend(cookie_manager.get_args())
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        elapsed = time.monotonic() - t0

        if proc.returncode == 0:
            size = video_path.stat().st_size if video_path.exists() else 0
            return DownloadResult(
                video_id=video_id, status="ok",
                file_size=size, elapsed=elapsed,
            )

        err_text = stderr.decode(errors="replace").strip()
        logger.warning(f"yt-dlp failed for {video_id}: {err_text}")
        category = classify_error(err_text)

        # Bot detection — try cookie escalation
        if category == "bot_blocked":
            escalated = cookie_manager.escalate()
            if escalated:
                # Retry once with new cookie strategy
                return await download_video(video_id, output_dir, cookie_manager, timeout)

        # Permanent errors are "skipped" (don't retry)
        if category == "permanent":
            return DownloadResult(
                video_id=video_id, status="skipped",
                error_category=category, elapsed=elapsed,
            )

        return DownloadResult(
            video_id=video_id, status="failed",
            error_category=category, elapsed=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        return DownloadResult(
            video_id=video_id, status="failed",
            error_category="timeout", elapsed=elapsed,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Unexpected error downloading {video_id}: {e}")
        return DownloadResult(
            video_id=video_id, status="failed",
            error_category="unknown", elapsed=elapsed,
        )
