import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg

logger = logging.getLogger(__name__)

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_NO_PROXY_HOSTS = (
    ".googleapis.com",
    ".google.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "huggingface.co",
)


def _looks_like_dead_loopback_proxy(proxy_url: str | None) -> bool:
    if not proxy_url:
        return False

    try:
        parsed = urlparse(proxy_url)
    except ValueError:
        return False

    return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9


def disable_dead_loopback_proxy() -> None:
    proxies = {
        key: value
        for key, value in ((key, os.getenv(key)) for key in _PROXY_ENV_KEYS)
        if value
    }
    if not proxies or not any(_looks_like_dead_loopback_proxy(value) for value in proxies.values()):
        return

    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    no_proxy_items = [item.strip() for item in no_proxy.split(",") if item.strip()]
    for host in _NO_PROXY_HOSTS:
        if host not in no_proxy_items:
            no_proxy_items.append(host)

    if no_proxy_items:
        joined = ",".join(no_proxy_items)
        os.environ["NO_PROXY"] = joined
        os.environ["no_proxy"] = joined

    logger.warning("Disabled dead loopback proxy settings for local runtime.")


def ensure_ffmpeg() -> str:
    ffmpeg_source = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    ffmpeg_dir = ffmpeg_source.parent

    path_parts = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    if str(ffmpeg_dir) not in path_parts:
        os.environ["PATH"] = os.pathsep.join([str(ffmpeg_dir), *path_parts])

    tools_dir = Path(".tools") / "ffmpeg"
    tools_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_alias = (tools_dir / "ffmpeg.exe").resolve()
    if not ffmpeg_alias.exists():
        shutil.copy2(ffmpeg_source, ffmpeg_alias)

    alias_dir = str(ffmpeg_alias.parent)
    path_parts = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    if alias_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([alias_dir, *path_parts])

    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_alias)
    os.environ["FFMPEG_BINARY"] = str(ffmpeg_alias)
    return str(ffmpeg_alias)


def prepare_runtime_environment() -> str:
    disable_dead_loopback_proxy()
    cache_root = Path(".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    huggingface_root = (cache_root / "huggingface").resolve()
    hub_cache_root = (huggingface_root / "hub").resolve()
    torch_root = (cache_root / "torch").resolve()
    nltk_root = (cache_root / "nltk").resolve()

    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_HOME"] = str(huggingface_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache_root)
    os.environ["TORCH_HOME"] = str(torch_root)
    os.environ["NLTK_DATA"] = str(nltk_root)
    return ensure_ffmpeg()
