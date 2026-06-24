"""Auto-bootstrap a bgutil PO-token provider for --political-videos.

PO tokens require a JS (BotGuard) runtime, so this needs node. If no provider is
already answering at the requested URL, we set one up ONCE (cached under
~/.cache/ytminer-client) and start it:
  * node: use system node>=20.19 if present, else download a standalone one
  * provider: git clone + build the bgutil provider (npm ships with node; needs git)
  * start it on the port and wait until it answers /ping

Returns the started subprocess (terminated on exit) or None (already running /
opted out / setup failed — the caller's canary then reports what's missing).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import click
import httpx

NODE_VER = "v22.22.3"
PROVIDER_BRANCH = "1.3.1"
PROVIDER_REPO = "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
NODE_PKGS = {
    ("Linux", "x86_64"): (f"node-{NODE_VER}-linux-x64", "tar.xz"),
    ("Linux", "aarch64"): (f"node-{NODE_VER}-linux-arm64", "tar.xz"),
    ("Darwin", "arm64"): (f"node-{NODE_VER}-darwin-arm64", "tar.gz"),
    ("Darwin", "x86_64"): (f"node-{NODE_VER}-darwin-x64", "tar.gz"),
}


def _ping(url: str) -> bool:
    try:
        httpx.get(url.rstrip("/") + "/ping", timeout=3)
        return True
    except Exception:
        return False


def _node_ok(node: str) -> bool:
    try:
        r = subprocess.run(
            [node, "-e", "const[a,b]=process.versions.node.split('.').map(Number);"
                         "process.stdout.write(String(a*100+b>=2019))"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "true"
    except Exception:
        return False


def _cached_node(cache: Path):
    return next((cache / "node").glob("node-*/bin/node"), None)


def _ensure_node(cache: Path) -> str:
    if shutil.which("node") and _node_ok("node"):
        return "node"
    c = _cached_node(cache)
    if c and _node_ok(str(c)):
        return str(c)
    key = (platform.system(), platform.machine())
    if key not in NODE_PKGS:
        raise RuntimeError(f"no prebuilt node for {key}; install node>=20.19 yourself")
    pkg, ext = NODE_PKGS[key]
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / f"node.{ext}"
    click.echo(f"  [provider] downloading node {NODE_VER} (one-time) ...")
    urllib.request.urlretrieve(f"https://nodejs.org/dist/{NODE_VER}/{pkg}.{ext}", tarball)
    ndir = cache / "node"
    shutil.rmtree(ndir, ignore_errors=True)
    ndir.mkdir(parents=True)
    with tarfile.open(tarball) as t:
        t.extractall(ndir)  # extracts to ndir/<pkg>/
    tarball.unlink(missing_ok=True)
    c = _cached_node(cache)
    if not c:
        raise RuntimeError("node download/extract failed")
    return str(c)


def _ensure_built(cache: Path, node: str) -> Path:
    pdir = cache / "bgutil-provider"
    main = pdir / "server" / "build" / "main.js"
    if main.exists():
        return main
    if not shutil.which("git"):
        raise RuntimeError("git is required to set up the PO-token provider")
    click.echo(f"  [provider] building bgutil provider {PROVIDER_BRANCH} (one-time, ~1-2 min) ...")
    shutil.rmtree(pdir, ignore_errors=True)
    subprocess.run(["git", "clone", "--quiet", "--single-branch", "--branch",
                    PROVIDER_BRANCH, PROVIDER_REPO, str(pdir)], check=True)
    env = {**os.environ, "PATH": str(Path(node).parent) + os.pathsep + os.environ.get("PATH", "")}
    server = pdir / "server"
    subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=server, check=True, env=env)
    subprocess.run(["npx", "tsc"], cwd=server, check=True, env=env)
    return main


def ensure_provider(provider_url: str, no_auto: bool = False):
    """Make sure a provider answers at provider_url; bootstrap+start one if not."""
    if _ping(provider_url):
        return None  # already running — nothing to manage
    if no_auto:
        return None  # caller's canary will explain what's missing
    port = provider_url.rstrip("/").rsplit(":", 1)[-1].split("/")[0]
    if not port.isdigit():
        port = "4416"
    cache = Path(os.environ.get("YTMINER_CACHE", Path.home() / ".cache" / "ytminer-client"))
    try:
        node = _ensure_node(cache)
        main = _ensure_built(cache, node)
    except Exception as e:
        click.echo(f"  [provider] auto-setup failed ({e}); run a bgutil provider yourself "
                   f"or pass --provider-url", err=True)
        return None
    log = open(cache / "provider.log", "a")
    proc = subprocess.Popen([node, str(main), "--port", str(port)],
                            stdout=log, stderr=log, start_new_session=True)
    for _ in range(40):
        time.sleep(1)
        if _ping(provider_url):
            click.echo(f"  [provider] running on port {port}")
            return proc
    click.echo(f"  [provider] did not come up; see {cache/'provider.log'}", err=True)
    return proc
