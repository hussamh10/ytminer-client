"""CLI for distributed YouTube video downloading."""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import time
from pathlib import Path

import click
import httpx

from ytminer_client import __version__
from ytminer_client.downloader import CookieManager, DownloadResult, RateLimiter, download_video

logger = logging.getLogger("ytminer-client")

COOLDOWN_STEPS = [5 * 60, 15 * 60, 30 * 60, 60 * 60]  # 5m, 15m, 30m, 1h


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


# ─── Server Communication ──────────────────────────────────────


class ServerClient:
    def __init__(self, server_url: str, worker_name: str):
        self.server_url = server_url.rstrip("/")
        self.worker_name = worker_name
        self.http = httpx.Client(timeout=30)

    def fetch_batch(self, channel: str | None = None, batch_size: int = 50) -> dict | None:
        params = {"worker": self.worker_name, "batch_size": batch_size}
        if channel:
            params["channel"] = channel
        resp = self.http.get(f"{self.server_url}/batch", params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("video_ids"):
            return None
        return data

    def report_batch(
        self,
        batch_id: str,
        results: dict[str, str],
        errors: dict[str, str],
        channel: str | None = None,
    ) -> dict:
        resp = self.http.post(
            f"{self.server_url}/report",
            json={
                "batch_id": batch_id,
                "worker": self.worker_name,
                "results": results,
                "errors": errors,
                "channel": channel,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def check_version(self) -> str | None:
        """Check if a newer client version is available. Returns new version or None."""
        try:
            resp = self.http.get(f"{self.server_url}/version", timeout=5)
            resp.raise_for_status()
            server_version = resp.json().get("client_version", "")
            if server_version and server_version != __version__:
                return server_version
        except Exception:
            pass
        return None

    def close(self):
        self.http.close()


# ─── Cooldown ──────────────────────────────────────────────────


async def bot_cooldown(
    cookie_manager: CookieManager,
    output_dir: Path,
    cooldown_step: int,
) -> bool:
    """Progressive cooldown on bot detection. Returns True if recovered."""
    wait_secs = COOLDOWN_STEPS[min(cooldown_step, len(COOLDOWN_STEPS) - 1)]
    wait_min = wait_secs // 60

    click.echo(f"  Bot detected! Cooling down for {wait_min}min...")
    await asyncio.sleep(wait_secs)

    # Test with a known public video
    click.echo(f"  Testing if block lifted...")
    test_result = await download_video(
        "dQw4w9WgXcQ", output_dir, cookie_manager, timeout=30,
    )

    if test_result.error_category == "bot_blocked":
        click.echo(f"  Still blocked.")
        return False

    click.echo(f"  Block lifted! Resuming downloads.")
    return True


# ─── Main Download Loop ─────────────────────────────────────────


async def download_loop(
    server: ServerClient,
    output_dir: Path,
    cookie_manager: CookieManager,
    rate_limiter: RateLimiter,
    channel: str | None,
    batch_size: int,
    update_check_interval: int = 5,
):
    """Main loop: fetch batch → download → report → repeat."""
    batch_count = 0
    session_ok = 0
    session_failed = 0
    session_skipped = 0
    session_start = time.monotonic()
    consecutive_bot = 0

    click.echo(f"Worker: {server.worker_name}")
    click.echo(f"Server: {server.server_url}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Cookie mode: {cookie_manager.mode}")
    click.echo()

    # First batch
    batch = server.fetch_batch(channel=channel, batch_size=batch_size)

    while batch:
        batch_id = batch["batch_id"]
        video_ids = batch["video_ids"]
        batch_channel = batch["channel"]
        batch_count += 1

        # Ensure channel output dir exists
        channel_dir = output_dir / batch_channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"--- Batch {batch_count} ({batch_channel}) | {len(video_ids)} videos ---")

        results: dict[str, str] = {}
        errors: dict[str, str] = {}

        for i, video_id in enumerate(video_ids):
            # Rate limit (skip delay for skipped/permanent videos)
            if i > 0 and not (results.get(video_ids[i - 1]) == "skipped"):
                await rate_limiter.wait()

            result = await download_video(video_id, channel_dir, cookie_manager)

            results[video_id] = result.status
            if result.error_category:
                errors[video_id] = result.error_category

            # Update session counters
            if result.status == "ok":
                session_ok += 1
                consecutive_bot = 0
                size_str = format_size(result.file_size)
                click.echo(f"  [{i+1}/{len(video_ids)}] {video_id}  OK  {size_str}  {result.elapsed:.1f}s")
                rate_limiter.report_success()
            elif result.status == "skipped":
                session_skipped += 1
                consecutive_bot = 0
                reason = result.error_category or "exists"
                click.echo(f"  [{i+1}/{len(video_ids)}] {video_id}  SKIP  ({reason})")
            else:
                session_failed += 1
                click.echo(f"  [{i+1}/{len(video_ids)}] {video_id}  FAIL  ({result.error_category})")
                rate_limiter.report_error(result.error_category or "unknown")

                # Bot detection cooldown
                if result.error_category == "bot_blocked":
                    consecutive_bot += 1
                    if consecutive_bot >= 3:
                        # Try progressive cooldowns
                        for step in range(len(COOLDOWN_STEPS)):
                            recovered = await bot_cooldown(cookie_manager, channel_dir, step)
                            if recovered:
                                consecutive_bot = 0
                                break
                        else:
                            click.echo("  All cooldowns exhausted. Stopping.")
                            # Report what we have so far
                            try:
                                server.report_batch(batch_id, results, errors, channel=channel)
                            except Exception:
                                pass
                            return
                else:
                    consecutive_bot = 0

        # Report and get next batch
        elapsed = time.monotonic() - session_start
        rate = session_ok / (elapsed / 3600) if elapsed > 60 else 0
        click.echo(
            f"  Batch done: {sum(1 for v in results.values() if v == 'ok')} ok, "
            f"{sum(1 for v in results.values() if v == 'failed')} failed, "
            f"{sum(1 for v in results.values() if v == 'skipped')} skipped"
        )
        click.echo(
            f"  Session total: {session_ok} ok, {session_failed} failed, "
            f"{session_skipped} skipped | {rate:.0f}/hr"
        )

        try:
            resp = server.report_batch(batch_id, results, errors, channel=channel)
            batch = resp.get("next_batch")
        except Exception as e:
            click.echo(f"  Error reporting to server: {e}")
            click.echo("  Retrying in 30s...")
            await asyncio.sleep(30)
            try:
                resp = server.report_batch(batch_id, results, errors, channel=channel)
                batch = resp.get("next_batch")
            except Exception as e2:
                click.echo(f"  Still failing: {e2}. Exiting.")
                break

        # Periodic version check
        if batch_count % update_check_interval == 0:
            new_version = server.check_version()
            if new_version:
                click.echo(
                    f"\n  Update available: v{__version__} → v{new_version}\n"
                    f"  Run: pip install --upgrade ytminer-client\n"
                )

    elapsed = time.monotonic() - session_start
    click.echo()
    click.echo(f"All done! {session_ok} downloaded, {session_failed} failed, {session_skipped} skipped")
    click.echo(f"Total time: {elapsed/3600:.1f}h")


# ─── CLI ─────────────────────────────────────────────────────────


@click.command()
@click.option("--server", required=True, help="Server URL (e.g. http://localhost:8000)")
@click.option("--output", default="./videos", help="Output directory for downloaded videos")
@click.option("--worker-name", default=None, help="Worker name (default: hostname)")
@click.option("--channel", default=None, help="Only download this channel (e.g. @geonews)")
@click.option("--batch-size", default=50, help="Videos per batch (default: 50)")
@click.option("--delay", default=30.0, help="Base delay between downloads in seconds (default: 30)")
@click.option("--jitter", default=10.0, help="Random jitter added to delay in seconds (default: 10)")
@click.option("--cookies", default=None, help="Path to cookies.txt file")
@click.option("--cookies-from-browser", default=None, help="Browser to extract cookies from (e.g. chrome, firefox)")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def main(server, output, worker_name, channel, batch_size, delay, jitter, cookies, cookies_from_browser, verbose):
    """Download YouTube videos from a ytminer server."""
    setup_logging(verbose)

    if worker_name is None:
        worker_name = platform.node() or "anonymous"

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cookie_manager = CookieManager(
        cookies_file=cookies,
        cookies_from_browser=cookies_from_browser,
    )
    rate_limiter = RateLimiter(base_delay=delay, jitter=jitter)

    srv = ServerClient(server, worker_name)

    # Quick connectivity check
    try:
        srv.http.get(f"{srv.server_url}/status", timeout=5).raise_for_status()
        click.echo(f"Connected to server: {server}")
    except Exception as e:
        click.echo(f"Cannot reach server at {server}: {e}", err=True)
        sys.exit(1)

    # Check for updates at startup
    new_version = srv.check_version()
    if new_version:
        click.echo(f"Update available: v{__version__} -> v{new_version}")
        click.echo(f"Run: pip install --upgrade ytminer-client")
        click.echo()

    try:
        asyncio.run(download_loop(
            server=srv,
            output_dir=output_dir,
            cookie_manager=cookie_manager,
            rate_limiter=rate_limiter,
            channel=channel,
            batch_size=batch_size,
        ))
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user. Goodbye!")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
