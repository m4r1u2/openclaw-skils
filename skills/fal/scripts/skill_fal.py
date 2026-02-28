#!/usr/bin/env python3
"""
OpenClaw SKILL script for fal.ai — full fal-client async API coverage.

Supports:
  - Text-to-Image (submit / subscribe / run / stream)
  - Image-to-Video (submit / subscribe / run / stream)
  - Async job management (status / result / cancel)
  - File & image upload (local file → fal CDN URL)
  - Base64 encoding (file / image / raw data → data URI)
  - Batch processing (concurrent async jobs)
  - Realtime WebSocket connections
  - Webhook delivery
  - Priority queuing & routing hints
  - Configurable timeouts (start_timeout, client_timeout)

Requirements:
  pip install fal-client
  export FAL_KEY="your-fal-api-key"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import fal_client
import httpx

# ─── Dynamic model discovery via fal.ai REST API ─────────────────────────────

FAL_MODELS_API = "https://fal.ai/api/models"
FAL_MODELS_PAGE_SIZE = 40


async def fetch_models(
    category: str | None = None,
    include_deprecated: bool = False,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch all models from the fal.ai public REST API (paginated).

    Args:
        category: Filter by category (e.g. "text-to-image", "image-to-video").
                  Filtering is done client-side (server param is unreliable).
        include_deprecated: If False (default), exclude deprecated models.
        search: Optional substring filter on model id/title.

    Returns:
        A list of model dicts with keys: id, title, category, shortDescription,
        deprecated, tags, etc.
    """
    all_items: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch first page to learn total page count
        resp = await client.get(FAL_MODELS_API, params={"page": 1})
        resp.raise_for_status()
        data = resp.json()
        all_items.extend(data.get("items", []))
        total_pages = data.get("pages", 1)

        # Fetch remaining pages concurrently in small batches
        for batch_start in range(2, total_pages + 1, 10):
            batch_end = min(batch_start + 10, total_pages + 1)
            tasks = []
            for page in range(batch_start, batch_end):
                tasks.append(client.get(FAL_MODELS_API, params={"page": page}))
            responses = await asyncio.gather(*tasks)
            for r in responses:
                r.raise_for_status()
                all_items.extend(r.json().get("items", []))

    # Client-side filtering
    if not include_deprecated:
        all_items = [m for m in all_items if not m.get("deprecated", False)]
    if category:
        cat_lower = category.lower()
        all_items = [m for m in all_items if (m.get("category") or "").lower() == cat_lower]
    if search:
        s = search.lower()
        all_items = [m for m in all_items if s in (m.get("id") or "").lower() or s in (m.get("title") or "").lower()]
    return all_items


# ─── Known model catalogs ────────────────────────────────────────────────────
# NOTE: These lists are for discoverability (list-t2i / list-i2v commands).
# The --model flag accepts ANY valid fal.ai endpoint — you are NOT limited
# to models listed here.  fal.ai adds new models frequently.

TEXT_TO_IMAGE_MODELS = [
    # Nano Banana / Gemini Flash Image
    "fal-ai/nano-banana-2",
    "fal-ai/nano-banana-pro",
    "fal-ai/gemini-3.1-flash-image-preview",
    # FLUX 2
    "fal-ai/flux-2-flex",
    # FLUX 1
    "fal-ai/flux/dev",
    "fal-ai/flux/schnell",
    "fal-ai/flux-pro/v1.1-ultra",
    "fal-ai/flux-pro/v1.1",
    "fal-ai/flux-pro",
    "fal-ai/flux-pro/new",
    "fal-ai/flux-realism",
    "fal-ai/flux-lora",
    "fal-ai/flux-lora-fill",
    "fal-ai/flux-differential-diffusion",
    "fal-ai/flux-krea-lora/stream",
    # Recraft
    "fal-ai/recraft-v3",
    "fal-ai/recraft/v3/text-to-image",
    "fal-ai/recraft/v4/text-to-image",
    "fal-ai/recraft/v4/pro/text-to-image",
    "fal-ai/recraft/v4/text-to-vector",
    "fal-ai/recraft/v4/pro/text-to-vector",
    # Stable Diffusion
    "fal-ai/stable-diffusion-v35-large",
    "fal-ai/stable-diffusion-v35-large/turbo",
    "fal-ai/stable-diffusion-v35-medium",
    "fal-ai/stable-cascade",
    # ByteDance / Seedream
    "fal-ai/bytedance/seedream/v5/lite/text-to-image",
    # BitDance
    "fal-ai/bitdance",
    # Qwen
    "fal-ai/qwen-image",
    # Ideogram
    "fal-ai/ideogram/v2",
    "fal-ai/ideogram/v2/turbo",
    # Bria
    "bria/fibo/generate",
    # ImagineArt
    "imagineart/imagineart-1.5-preview/text-to-image",
    # Other
    "fal-ai/aura-flow",
    "fal-ai/kolors",
    "fal-ai/fooocus",
]

IMAGE_TO_VIDEO_MODELS = [
    # Veo 3.1 (Google)
    "fal-ai/veo3.1/image-to-video",
    "fal-ai/veo3.1/fast/image-to-video",
    "fal-ai/veo3.1/reference-to-video",
    "fal-ai/veo3.1/first-last-frame-to-video",
    "fal-ai/veo3.1/fast/first-last-frame-to-video",
    # Sora 2 (OpenAI)
    "fal-ai/sora-2/image-to-video",
    "fal-ai/sora-2/image-to-video/pro",
    # Kling
    "fal-ai/kling-video/o3/standard/image-to-video",
    "fal-ai/kling-video/v3/pro/image-to-video",
    "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
    "fal-ai/kling-video/v2.1/master/image-to-video",
    "fal-ai/kling-video/v2.1/pro/image-to-video",
    "fal-ai/kling-video/v2/master/image-to-video",
    "fal-ai/kling-video/v1.5/pro/image-to-video",
    "fal-ai/kling-video/v1/pro/image-to-video",
    "fal-ai/kling-video/v1/standard/image-to-video",
    # MiniMax / Hailuo
    "fal-ai/minimax-video/image-to-video",
    "fal-ai/minimax/hailuo-02/standard/image-to-video",
    # Runway
    "fal-ai/runway-gen3/turbo/image-to-video",
    # LTX
    "fal-ai/ltx-2-19b/image-to-video",
    "fal-ai/ltx-video-13b-distilled/image-to-video",
    # Wan
    "fal-ai/wan/v2.1/image-to-video",
    "fal-ai/wan/v2.2-a14b/image-to-video",
    "fal-ai/wan/v2.2-a14b/image-to-video/lora",
    # PixVerse
    "fal-ai/pixverse/v5/image-to-video",
    # Cosmos (NVIDIA)
    "fal-ai/cosmos-predict-2.5/image-to-video",
    # Vidu
    "fal-ai/vidu/q3/image-to-video/turbo",
    # Lucy (Decart)
    "decart/lucy-i2v",
    "decart/lucy-14b/image-to-video",
    # Luma
    "fal-ai/luma-dream-machine/image-to-video",
    # HunyuanVideo
    "fal-ai/hunyuan-video/image-to-video",
    # CogVideoX
    "fal-ai/cogvideox-5b/image-to-video",
    # Stable Video
    "fal-ai/stable-video",
    # HeyGen
    "fal-ai/heygen/avatar4/image-to-video",
    # Creatify
    "fal-ai/creatify/aurora",
    # VEED
    "veed/fabric-1.0",
    # Omnihuman
    "fal-ai/bytedance/omnihuman/v1.5",
    # MultiTalk
    "fal-ai/ai-avatar/single-text",
]


# ─── Queue progress callback ─────────────────────────────────────────────────


def _on_queue_update(update: fal_client.Status) -> None:
    """Print live queue / progress updates to stderr."""
    if isinstance(update, fal_client.Queued):
        print(f"[queue] Position: {update.position}", file=sys.stderr)
    elif isinstance(update, fal_client.InProgress):
        logs = getattr(update, "logs", None)
        if logs:
            for log in logs:
                msg = log.get("message", "") if isinstance(log, dict) else str(log)
                print(f"[progress] {msg}", file=sys.stderr)
    elif isinstance(update, fal_client.Completed):
        logs = getattr(update, "logs", None)
        if logs:
            for log in logs:
                msg = log.get("message", "") if isinstance(log, dict) else str(log)
                print(f"[done] {msg}", file=sys.stderr)


# ─── Argument builders ───────────────────────────────────────────────────────


def _build_t2i_arguments(
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    seed: int | None = None,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_images": num_images,
    }
    if negative_prompt:
        args["negative_prompt"] = negative_prompt
    if seed is not None:
        args["seed"] = seed
    if guidance_scale is not None:
        args["guidance_scale"] = guidance_scale
    if num_inference_steps is not None:
        args["num_inference_steps"] = num_inference_steps
    if extra:
        args.update(extra)
    return args


def _build_i2v_arguments(
    image_url: str,
    *,
    prompt: str | None = None,
    duration: float | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"image_url": image_url}
    if prompt:
        args["prompt"] = prompt
    if duration is not None:
        args["duration"] = duration
    if seed is not None:
        args["seed"] = seed
    if extra:
        args.update(extra)
    return args


# ─── Core async operations ───────────────────────────────────────────────────


async def submit_job(
    model: str,
    arguments: dict[str, Any],
    *,
    path: str = "",
    hint: str | None = None,
    webhook_url: str | None = None,
    priority: str | None = None,
    start_timeout: float | None = None,
) -> str:
    """Submit a job to the queue. Returns request_id immediately."""
    kwargs: dict[str, Any] = {}
    if path:
        kwargs["path"] = path
    if hint:
        kwargs["hint"] = hint
    if webhook_url:
        kwargs["webhook_url"] = webhook_url
    if priority:
        kwargs["priority"] = priority
    if start_timeout is not None:
        kwargs["start_timeout"] = start_timeout

    handle = await fal_client.submit_async(model, arguments, **kwargs)
    return handle.request_id


async def subscribe_job(
    model: str,
    arguments: dict[str, Any],
    *,
    path: str = "",
    hint: str | None = None,
    priority: str | None = None,
    start_timeout: float | None = None,
    client_timeout: float | None = None,
) -> dict[str, Any]:
    """Subscribe to a job — blocks with live progress until done."""
    kwargs: dict[str, Any] = {
        "with_logs": True,
        "on_queue_update": _on_queue_update,
    }
    if path:
        kwargs["path"] = path
    if hint:
        kwargs["hint"] = hint
    if priority:
        kwargs["priority"] = priority
    if start_timeout is not None:
        kwargs["start_timeout"] = start_timeout
    if client_timeout is not None:
        kwargs["client_timeout"] = client_timeout

    return await fal_client.subscribe_async(model, arguments, **kwargs)


async def run_job(
    model: str,
    arguments: dict[str, Any],
    *,
    path: str = "",
    hint: str | None = None,
    timeout: float | None = None,
    start_timeout: float | None = None,
) -> dict[str, Any]:
    """Direct run — single HTTP call, no queue. Best for fast models."""
    kwargs: dict[str, Any] = {}
    if path:
        kwargs["path"] = path
    if hint:
        kwargs["hint"] = hint
    if timeout is not None:
        kwargs["timeout"] = timeout
    if start_timeout is not None:
        kwargs["start_timeout"] = start_timeout

    return await fal_client.run_async(model, arguments, **kwargs)


async def stream_job(
    model: str,
    arguments: dict[str, Any],
    *,
    path: str = "/stream",
    timeout: float | None = None,
) -> None:
    """Stream server-sent events from a model. Prints each chunk as JSON."""
    kwargs: dict[str, Any] = {"path": path}
    if timeout is not None:
        kwargs["timeout"] = timeout

    async for event in fal_client.stream_async(model, arguments, **kwargs):
        print(json.dumps(event, default=str))


# ─── Job management ──────────────────────────────────────────────────────────


async def check_status(model: str, request_id: str) -> fal_client.Status:
    return await fal_client.status_async(model, request_id, with_logs=True)


async def fetch_result(model: str, request_id: str) -> dict[str, Any]:
    return await fal_client.result_async(model, request_id)


async def cancel_job(model: str, request_id: str) -> None:
    await fal_client.cancel_async(model, request_id)
    print(f"Cancelled {request_id}", file=sys.stderr)


# ─── Upload helpers ───────────────────────────────────────────────────────────


async def upload_file(file_path: str) -> str:
    """Upload a local file to fal CDN and return the URL."""
    return await fal_client.upload_file_async(Path(file_path))


async def upload_image(file_path: str, fmt: str = "jpeg") -> str:
    """Upload a local image (via PIL) to fal CDN and return the URL."""
    from PIL import Image

    img = Image.open(file_path)
    return await fal_client.upload_image_async(img, format=fmt)


async def upload_raw(data_path: str, content_type: str, file_name: str | None = None) -> str:
    """Upload raw bytes from a file to fal CDN and return the URL."""
    raw = Path(data_path).read_bytes()
    return await fal_client.upload_async(raw, content_type, file_name)


# ─── Encode helpers (local, no network) ──────────────────────────────────────


def encode_file(file_path: str) -> str:
    """Base64-encode a local file as a data URI string."""
    return fal_client.encode_file(Path(file_path))


def encode_image(file_path: str, fmt: str = "jpeg") -> str:
    """Base64-encode a local image (via PIL) as a data URI string."""
    from PIL import Image

    img = Image.open(file_path)
    return fal_client.encode_image(img, format=fmt)


def encode_raw(data_path: str, content_type: str) -> str:
    """Base64-encode raw bytes as a data URI string."""
    raw = Path(data_path).read_bytes()
    return fal_client.encode(raw, content_type)


# ─── Realtime (WebSocket) ────────────────────────────────────────────────────


async def realtime_session(
    model: str,
    arguments: dict[str, Any],
    *,
    path: str = "/realtime",
    num_rounds: int = 1,
) -> None:
    """
    Open a realtime WebSocket connection, send arguments, and print responses.
    Useful for interactive / low-latency models (e.g. turbo-diffusion).
    """
    async with fal_client.realtime_async(model, path=path) as conn:
        for i in range(num_rounds):
            await conn.send(arguments)
            response = await conn.recv()
            if response is not None:
                print(json.dumps(response, default=str))
            else:
                print(f"[realtime] Round {i + 1}: no response", file=sys.stderr)


# ─── Batch helpers ────────────────────────────────────────────────────────────


async def batch_text_to_image(
    model: str,
    prompts: list[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run multiple text-to-image prompts concurrently via asyncio.gather."""
    tasks = []
    for prompt in prompts:
        args = _build_t2i_arguments(prompt, **kwargs)
        tasks.append(subscribe_job(model, args))
    return await asyncio.gather(*tasks)


async def batch_image_to_video(
    model: str,
    image_urls: list[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run multiple image-to-video jobs concurrently via asyncio.gather."""
    tasks = []
    for url in image_urls:
        args = _build_i2v_arguments(url, **kwargs)
        tasks.append(subscribe_job(model, args))
    return await asyncio.gather(*tasks)


# ─── Pretty output ────────────────────────────────────────────────────────────


def _print_json(result: Any) -> None:
    print(json.dumps(result, indent=2, default=str))


def _print_status(status: fal_client.Status) -> None:
    if isinstance(status, fal_client.Queued):
        print(f"Status: QUEUED (position {status.position})")
    elif isinstance(status, fal_client.InProgress):
        print("Status: IN_PROGRESS")
    elif isinstance(status, fal_client.Completed):
        print("Status: COMPLETED")
    else:
        print(f"Status: {status}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    """Flags shared by all generation commands."""
    p.add_argument("--model", required=True, help="fal.ai model endpoint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--extra-json", type=str, default=None, help="Extra model arguments as a JSON string")
    p.add_argument("--path", type=str, default="", help="Sub-path on the model endpoint")
    p.add_argument("--hint", type=str, default=None, help="Routing hint for the fal queue")
    p.add_argument("--priority", type=str, default=None, help="Queue priority (e.g. 'normal', 'rush')")
    p.add_argument("--start-timeout", type=float, default=None, help="Server-side start timeout in seconds")
    p.add_argument("--client-timeout", type=float, default=None, help="Client-side HTTP timeout in seconds")


def _add_t2i_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)


def _add_i2v_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image-url", required=True, help="URL (or local path to auto-upload) of the source image")
    p.add_argument("--prompt", default=None)
    p.add_argument("--duration", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenClaw SKILL — fal.ai text-to-image & image-to-video (full async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── list models ──────────────────────────────────────────────────────
    sub.add_parser("list-t2i", help="List known text-to-image models (hardcoded)")
    sub.add_parser("list-i2v", help="List known image-to-video models (hardcoded)")

    p = sub.add_parser("list-models", help="Fetch models dynamically from fal.ai API")
    p.add_argument(
        "--category",
        default=None,
        help="Filter by category (e.g. text-to-image, image-to-video, "
        "text-to-video, image-to-image, video-to-video, training)",
    )
    p.add_argument("--search", default=None, help="Substring search on model id or title")
    p.add_argument("--include-deprecated", action="store_true", default=False, help="Include deprecated models")
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=False,
        help="Output full JSON details instead of one-id-per-line",
    )
    p.add_argument(
        "--categories",
        action="store_true",
        default=False,
        help="List all available categories and their counts, then exit",
    )

    # ── submit (fire-and-forget → request_id) ────────────────────────────
    p = sub.add_parser("submit-t2i", help="Submit async text-to-image job")
    _add_common(p)
    _add_t2i_args(p)
    p.add_argument("--webhook-url", default=None)

    p = sub.add_parser("submit-i2v", help="Submit async image-to-video job")
    _add_common(p)
    _add_i2v_args(p)
    p.add_argument("--webhook-url", default=None)

    # ── subscribe (wait with progress) ───────────────────────────────────
    p = sub.add_parser("subscribe-t2i", help="Subscribe text-to-image (wait for result)")
    _add_common(p)
    _add_t2i_args(p)

    p = sub.add_parser("subscribe-i2v", help="Subscribe image-to-video (wait for result)")
    _add_common(p)
    _add_i2v_args(p)

    # ── run (direct HTTP, no queue) ──────────────────────────────────────
    p = sub.add_parser("run-t2i", help="Direct-run text-to-image (no queue)")
    _add_common(p)
    _add_t2i_args(p)

    p = sub.add_parser("run-i2v", help="Direct-run image-to-video (no queue)")
    _add_common(p)
    _add_i2v_args(p)

    # ── stream (SSE) ─────────────────────────────────────────────────────
    p = sub.add_parser("stream-t2i", help="Stream text-to-image via SSE")
    _add_common(p)
    _add_t2i_args(p)

    p = sub.add_parser("stream-i2v", help="Stream image-to-video via SSE")
    _add_common(p)
    _add_i2v_args(p)

    # ── job management ───────────────────────────────────────────────────
    for cmd, hlp in [
        ("status", "Check job status"),
        ("result", "Fetch job result"),
        ("cancel", "Cancel a running/queued job"),
    ]:
        p = sub.add_parser(cmd, help=hlp)
        p.add_argument("--model", required=True)
        p.add_argument("--request-id", required=True)

    # ── upload ───────────────────────────────────────────────────────────
    p = sub.add_parser("upload-file", help="Upload a local file to fal CDN")
    p.add_argument("--file", required=True, help="Path to the local file")

    p = sub.add_parser("upload-image", help="Upload a local image (PIL) to fal CDN")
    p.add_argument("--file", required=True, help="Path to the local image")
    p.add_argument("--format", default="jpeg", help="Image format (jpeg, png, webp)")

    p = sub.add_parser("upload-raw", help="Upload raw bytes to fal CDN")
    p.add_argument("--file", required=True, help="Path to the file to read bytes from")
    p.add_argument("--content-type", required=True, help="MIME content type")
    p.add_argument("--file-name", default=None, help="Optional file name")

    # ── encode (local base64, no network) ────────────────────────────────
    p = sub.add_parser("encode-file", help="Base64-encode a local file as data URI")
    p.add_argument("--file", required=True)

    p = sub.add_parser("encode-image", help="Base64-encode a local image as data URI")
    p.add_argument("--file", required=True)
    p.add_argument("--format", default="jpeg")

    p = sub.add_parser("encode-raw", help="Base64-encode raw bytes as data URI")
    p.add_argument("--file", required=True)
    p.add_argument("--content-type", required=True)

    # ── realtime (WebSocket) ─────────────────────────────────────────────
    p = sub.add_parser("realtime", help="Open a realtime WebSocket session")
    _add_common(p)
    p.add_argument("--arguments-json", required=True, help="JSON string of arguments to send each round")
    p.add_argument("--rounds", type=int, default=1, help="Number of send/recv rounds")

    # ── batch ────────────────────────────────────────────────────────────
    p = sub.add_parser("batch-t2i", help="Batch concurrent text-to-image jobs")
    _add_common(p)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)

    p = sub.add_parser("batch-i2v", help="Batch concurrent image-to-video jobs")
    _add_common(p)
    p.add_argument("--image-urls", nargs="+", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--duration", type=float, default=None)

    return parser


# ─── Auto-upload helper ──────────────────────────────────────────────────────


async def _maybe_upload(image_url: str) -> str:
    """If image_url is a local path, upload it first and return the CDN URL."""
    if os.path.isfile(image_url):
        print(f"[upload] Uploading local file: {image_url}", file=sys.stderr)
        return await upload_file(image_url)
    return image_url


# ─── Main dispatch ────────────────────────────────────────────────────────────


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    extra = json.loads(args.extra_json) if getattr(args, "extra_json", None) else None

    # Convenience extractors for shared flags
    def common_submit_kw() -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if getattr(args, "path", ""):
            kw["path"] = args.path
        if getattr(args, "hint", None):
            kw["hint"] = args.hint
        if getattr(args, "priority", None):
            kw["priority"] = args.priority
        if getattr(args, "start_timeout", None) is not None:
            kw["start_timeout"] = args.start_timeout
        if getattr(args, "webhook_url", None):
            kw["webhook_url"] = args.webhook_url
        return kw

    def common_subscribe_kw() -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if getattr(args, "path", ""):
            kw["path"] = args.path
        if getattr(args, "hint", None):
            kw["hint"] = args.hint
        if getattr(args, "priority", None):
            kw["priority"] = args.priority
        if getattr(args, "start_timeout", None) is not None:
            kw["start_timeout"] = args.start_timeout
        if getattr(args, "client_timeout", None) is not None:
            kw["client_timeout"] = args.client_timeout
        return kw

    def common_run_kw() -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if getattr(args, "path", ""):
            kw["path"] = args.path
        if getattr(args, "hint", None):
            kw["hint"] = args.hint
        if getattr(args, "start_timeout", None) is not None:
            kw["start_timeout"] = args.start_timeout
        if getattr(args, "client_timeout", None) is not None:
            kw["timeout"] = args.client_timeout
        return kw

    match args.command:
        # ── List models ──────────────────────────────────────────────────
        case "list-t2i":
            for m in TEXT_TO_IMAGE_MODELS:
                print(m)
        case "list-i2v":
            for m in IMAGE_TO_VIDEO_MODELS:
                print(m)

        case "list-models":
            if args.categories:
                # Fetch all models and print category counts
                models = await fetch_models(include_deprecated=args.include_deprecated)
                cats: dict[str, int] = {}
                for m in models:
                    c = m.get("category") or "unknown"
                    cats[c] = cats.get(c, 0) + 1
                for c, n in sorted(cats.items(), key=lambda x: -x[1]):
                    print(f"{c}: {n}")
            else:
                models = await fetch_models(
                    category=args.category,
                    include_deprecated=args.include_deprecated,
                    search=args.search,
                )
                if args.as_json:
                    _print_json(models)
                else:
                    for m in models:
                        dep = " [DEPRECATED]" if m.get("deprecated") else ""
                        cat = m.get("category") or ""
                        title = m.get("title") or ""
                        print(f"{m['id']}  ({cat})  {title}{dep}")

        # ── Submit ───────────────────────────────────────────────────────
        case "submit-t2i":
            t2i = _build_t2i_arguments(
                args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                extra=extra,
            )
            rid = await submit_job(args.model, t2i, **common_submit_kw())
            _print_json({"request_id": rid, "model": args.model})

        case "submit-i2v":
            url = await _maybe_upload(args.image_url)
            i2v = _build_i2v_arguments(
                url,
                prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
                extra=extra,
            )
            rid = await submit_job(args.model, i2v, **common_submit_kw())
            _print_json({"request_id": rid, "model": args.model})

        # ── Subscribe ────────────────────────────────────────────────────
        case "subscribe-t2i":
            t2i = _build_t2i_arguments(
                args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                extra=extra,
            )
            _print_json(await subscribe_job(args.model, t2i, **common_subscribe_kw()))

        case "subscribe-i2v":
            url = await _maybe_upload(args.image_url)
            i2v = _build_i2v_arguments(
                url,
                prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
                extra=extra,
            )
            _print_json(await subscribe_job(args.model, i2v, **common_subscribe_kw()))

        # ── Run (direct, no queue) ───────────────────────────────────────
        case "run-t2i":
            t2i = _build_t2i_arguments(
                args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                extra=extra,
            )
            _print_json(await run_job(args.model, t2i, **common_run_kw()))

        case "run-i2v":
            url = await _maybe_upload(args.image_url)
            i2v = _build_i2v_arguments(
                url,
                prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
                extra=extra,
            )
            _print_json(await run_job(args.model, i2v, **common_run_kw()))

        # ── Stream (SSE) ─────────────────────────────────────────────────
        case "stream-t2i":
            t2i = _build_t2i_arguments(
                args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                extra=extra,
            )
            await stream_job(
                args.model,
                t2i,
                path=args.path or "/stream",
                timeout=args.client_timeout,
            )

        case "stream-i2v":
            url = await _maybe_upload(args.image_url)
            i2v = _build_i2v_arguments(
                url,
                prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
                extra=extra,
            )
            await stream_job(
                args.model,
                i2v,
                path=args.path or "/stream",
                timeout=args.client_timeout,
            )

        # ── Job management ───────────────────────────────────────────────
        case "status":
            _print_status(await check_status(args.model, args.request_id))
        case "result":
            _print_json(await fetch_result(args.model, args.request_id))
        case "cancel":
            await cancel_job(args.model, args.request_id)
            print("OK")

        # ── Upload ───────────────────────────────────────────────────────
        case "upload-file":
            url = await upload_file(args.file)
            print(url)
        case "upload-image":
            url = await upload_image(args.file, fmt=args.format)
            print(url)
        case "upload-raw":
            url = await upload_raw(args.file, args.content_type, args.file_name)
            print(url)

        # ── Encode (local, no network) ───────────────────────────────────
        case "encode-file":
            print(encode_file(args.file))
        case "encode-image":
            print(encode_image(args.file, fmt=args.format))
        case "encode-raw":
            print(encode_raw(args.file, args.content_type))

        # ── Realtime (WebSocket) ─────────────────────────────────────────
        case "realtime":
            rt_args = json.loads(args.arguments_json)
            await realtime_session(
                args.model,
                rt_args,
                path=args.path or "/realtime",
                num_rounds=args.rounds,
            )

        # ── Batch ────────────────────────────────────────────────────────
        case "batch-t2i":
            results = await batch_text_to_image(
                args.model,
                args.prompts,
                negative_prompt=getattr(args, "negative_prompt", None),
                width=args.width,
                height=args.height,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                extra=extra,
            )
            for i, r in enumerate(results):
                print(f"\n── Prompt {i + 1} ──", file=sys.stderr)
                _print_json(r)

        case "batch-i2v":
            urls = [await _maybe_upload(u) for u in args.image_urls]
            results = await batch_image_to_video(
                args.model,
                urls,
                prompt=getattr(args, "prompt", None),
                duration=getattr(args, "duration", None),
                seed=args.seed,
                extra=extra,
            )
            for i, r in enumerate(results):
                print(f"\n── Image {i + 1} ──", file=sys.stderr)
                _print_json(r)

        case _:
            parser.print_help()
            sys.exit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
