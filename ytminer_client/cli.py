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

COOLDOWN_STEPS = [30 * 60, 60 * 60, 2 * 3600]  # 30m, 1h, 2h (then 2h forever)


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


# ─── Uploader ─────────────────────────────────────────────────


class Uploader:
    def __init__(self, server_url: str, worker_name: str, enabled: bool = False, cleanup: bool = True):
        self.server_url = server_url.rstrip("/")
        self.worker_name = worker_name
        self.enabled = enabled
        self.cleanup = cleanup
        self._http: httpx.AsyncClient | None = None
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.total_uploaded = 0
        self.total_bytes = 0
        self.pending = 0

    async def start(self):
        if self.enabled:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._worker())

    async def close(self):
        if self._queue and self._task:
            await self._queue.join()  # wait for pending uploads to finish
            self._task.cancel()
        if self._http:
            await self._http.aclose()

    def enqueue(self, video_id: str, channel: str, channel_dir: Path):
        """Queue a video for background upload. Non-blocking."""
        if self.enabled and self._queue is not None:
            self._queue.put_nowait((video_id, channel, channel_dir))
            self.pending += 1

    async def _worker(self):
        """Background task that drains the upload queue."""
        while True:
            try:
                video_id, channel, channel_dir = await self._queue.get()
                await self._upload_video(video_id, channel, channel_dir)
                self._queue.task_done()
                self.pending -= 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Upload worker error: {e}")
                self._queue.task_done()
                self.pending -= 1

    async def _upload_video(self, video_id: str, channel: str, channel_dir: Path):
        video_path = channel_dir / f"{video_id}.mp4"
        info_path = channel_dir / f"{video_id}.info.json"

        success = True
        for path in [video_path, info_path]:
            if not path.exists():
                continue
            ok = await self._upload_file(channel, video_id, path)
            if not ok:
                success = False

        if success and self.cleanup:
            for path in [video_path, info_path]:
                if path.exists():
                    path.unlink()

        if success:
            self.total_uploaded += 1

    @staticmethod
    async def _file_chunks(file_path: Path, chunk_size: int = 1024 * 1024):
        """Async generator that yields file contents in chunks for streaming upload."""
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def _upload_file(self, channel: str, video_id: str, file_path: Path) -> bool:
        url = f"{self.server_url}/upload/{channel}/{video_id}"
        file_size = file_path.stat().st_size
        # Scale timeout with file size: 60s base + 1s per MB, min 120s
        timeout_secs = max(120, 60 + file_size // (1024 * 1024))
        wait = 30
        for attempt in range(5):
            try:
                # Stream via async generator — passing a sync file object to
                # AsyncClient triggers "sync request with AsyncClient" error.
                resp = await self._http.post(
                    url,
                    content=self._file_chunks(file_path),
                    params={"worker": self.worker_name, "filename": file_path.name},
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(file_size),
                    },
                    timeout=httpx.Timeout(timeout_secs, connect=10.0),
                )
                resp.raise_for_status()
                self.total_bytes += file_size
                return True
            except Exception as e:
                logger.warning(
                    f"Upload failed for {file_path.name} "
                    f"({format_size(file_size)}, attempt {attempt+1}/5): {e}"
                )
                if attempt < 4:
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 300)
        logger.error(f"Giving up on {file_path.name} after 5 attempts — skipping")
        return False


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


# ─── Network Retry Helpers ─────────────────────────────────────


async def fetch_with_retry(server: ServerClient, channel: str | None, batch_size: int) -> dict | None:
    """Fetch a batch, retrying forever on network errors."""
    wait = 30
    while True:
        try:
            batch = server.fetch_batch(channel=channel, batch_size=batch_size)
            return batch
        except Exception as e:
            click.echo(f"  Network error fetching batch: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 600)  # cap at 10min


async def report_with_retry(
    server: ServerClient, batch_id: str, results: dict, errors: dict, channel: str | None,
) -> dict:
    """Report batch results, retrying forever on network errors."""
    wait = 30
    while True:
        try:
            return server.report_batch(batch_id, results, errors, channel=channel)
        except Exception as e:
            click.echo(f"  Network error reporting batch: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 600)


# ─── Main Download Loop ─────────────────────────────────────────


async def download_loop(
    server: ServerClient,
    output_dir: Path,
    cookie_manager: CookieManager,
    rate_limiter: RateLimiter,
    uploader: Uploader,
    channel: str | None,
    batch_size: int,
    update_check_interval: int = 5,
):
    """Main loop: fetch batch → download → upload → report → repeat."""
    batch_count = 0
    session_ok = 0
    session_failed = 0
    session_skipped = 0
    session_start = time.monotonic()
    consecutive_bot = 0

    # One-time async browser detection (non-blocking)
    await cookie_manager.warmup()

    # Start uploader if enabled
    await uploader.start()

    click.echo(f"Worker: {server.worker_name}")
    click.echo(f"Server: {server.server_url}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Cookie mode: {cookie_manager.mode}")
    if uploader.enabled:
        click.echo(f"Upload: enabled (cleanup: {uploader.cleanup})")
    click.echo()

    # First batch
    batch = await fetch_with_retry(server, channel, batch_size)

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

                # Queue for background upload
                if uploader.enabled:
                    uploader.enqueue(video_id, batch_channel, channel_dir)
                    click.echo(f"    -> queued for upload ({uploader.pending} pending)")

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
                        step = 0
                        while True:
                            recovered = await bot_cooldown(cookie_manager, channel_dir, step)
                            if recovered:
                                consecutive_bot = 0
                                break
                            step += 1
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

        # Wait for uploads to finish before reporting (so 'done' = actually uploaded)
        if uploader.enabled and uploader._queue:
            await uploader._queue.join()

        resp = await report_with_retry(server, batch_id, results, errors, channel)
        batch = resp.get("next_batch")
        if not batch:
            batch = await fetch_with_retry(server, channel, batch_size)

        # Periodic version check
        if batch_count % update_check_interval == 0:
            new_version = server.check_version()
            if new_version:
                click.echo(
                    f"\n  Update available: v{__version__} -> v{new_version}\n"
                    f"  Run: pip install --upgrade ytminer-client\n"
                )

    elapsed = time.monotonic() - session_start
    click.echo()
    click.echo(f"All done! {session_ok} downloaded, {session_failed} failed, {session_skipped} skipped")
    click.echo(f"Total time: {elapsed/3600:.1f}h")

    await uploader.close()


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
@click.option("--upload", is_flag=True, help="Upload videos to server after download (for ephemeral envs like Colab)")
@click.option("--keep-files", is_flag=True, help="Keep local files after upload (default: delete after upload)")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def main(server, output, worker_name, channel, batch_size, delay, jitter, cookies, cookies_from_browser, upload, keep_files, verbose):
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
    uploader = Uploader(
        server_url=server,
        worker_name=worker_name,
        enabled=upload,
        cleanup=not keep_files,
    )

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
            uploader=uploader,
            channel=channel,
            batch_size=batch_size,
        ))
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user. Goodbye!")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
