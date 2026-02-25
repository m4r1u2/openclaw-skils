#!/usr/bin/env python3
"""
Kling AI Video Generation CLI

Generates videos using the Kling AI API.
Supports text-to-video, image-to-video, and video extension.

Requirements: pip install PyJWT requests

Environment variables:
  KLING_ACCESS_KEY  — Your Kling AI access key
  KLING_SECRET_KEY  — Your Kling AI secret key
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

try:
    import jwt
except ImportError:
    print("Error: PyJWT not installed. Run: pip install PyJWT", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://api-singapore.klingai.com"

# ── Authentication ──────────────────────────────────────────────────

def get_jwt_token():
    """Generate a JWT token from KLING_ACCESS_KEY and KLING_SECRET_KEY."""
    access_key = os.environ.get("KLING_ACCESS_KEY")
    secret_key = os.environ.get("KLING_SECRET_KEY")

    if not access_key or not secret_key:
        print("Error: KLING_ACCESS_KEY and KLING_SECRET_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": access_key,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, secret_key, headers=headers)


def api_headers():
    """Return authorization headers."""
    return {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json",
    }


# ── API Calls ───────────────────────────────────────────────────────

def create_text2video(args):
    """Create a text-to-video generation task."""
    body = {
        "model_name": args.model,
        "prompt": args.prompt,
        "mode": args.mode,
        "duration": args.duration,
        "aspect_ratio": args.aspect_ratio,
    }

    if args.negative_prompt:
        body["negative_prompt"] = args.negative_prompt

    if args.sound and args.sound != "off":
        body["sound"] = args.sound

    if args.cfg_scale is not None:
        body["cfg_scale"] = args.cfg_scale

    if args.camera:
        body["camera_control"] = {"type": args.camera}

    url = f"{BASE_URL}/v1/videos/text2video"
    print(f"Creating text-to-video task...")
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode}")
    print(f"  Duration: {args.duration}s")
    print(f"  Aspect ratio: {args.aspect_ratio}")
    print(f"  Prompt: {args.prompt[:100]}...")
    print()

    resp = requests.post(url, headers=api_headers(), json=body)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        print(f"API Error: {data.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    task_id = data["data"]["task_id"]
    print(f"Task created: {task_id}")
    return task_id, "text2video"


def create_image2video(args):
    """Create an image-to-video generation task."""
    image_data = args.image
    if os.path.isfile(image_data):
        with open(image_data, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    elif image_data.startswith("http"):
        pass  # URL — send as-is
    # else assume base64

    body = {
        "model_name": args.model,
        "image": image_data,
        "mode": args.mode,
        "duration": args.duration,
        "aspect_ratio": args.aspect_ratio,
    }

    if args.prompt:
        body["prompt"] = args.prompt
    if args.negative_prompt:
        body["negative_prompt"] = args.negative_prompt
    if args.image_tail:
        tail = args.image_tail
        if os.path.isfile(tail):
            with open(tail, "rb") as f:
                tail = base64.b64encode(f.read()).decode("utf-8")
        body["image_tail"] = tail
    if args.sound and args.sound != "off":
        body["sound"] = args.sound
    if args.cfg_scale is not None:
        body["cfg_scale"] = args.cfg_scale

    url = f"{BASE_URL}/v1/videos/image2video"
    print(f"Creating image-to-video task...")
    resp = requests.post(url, headers=api_headers(), json=body)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        print(f"API Error: {data.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    task_id = data["data"]["task_id"]
    print(f"Task created: {task_id}")
    return task_id, "image2video"


def create_extend(args):
    """Extend an existing video."""
    body = {
        "model_name": args.model,
        "task_id": args.task_id,
        "duration": args.duration,
    }
    if args.prompt:
        body["prompt"] = args.prompt

    url = f"{BASE_URL}/v1/videos/video-extend"
    print(f"Creating video extension task from {args.task_id}...")
    resp = requests.post(url, headers=api_headers(), json=body)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        print(f"API Error: {data.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    task_id = data["data"]["task_id"]
    print(f"Extension task created: {task_id}")
    return task_id, "video-extend"


# ── Polling & Download ──────────────────────────────────────────────

def poll_task(task_id, endpoint_type, timeout=900, interval=5):
    """Poll task status until complete or timeout."""
    url = f"{BASE_URL}/v1/videos/{endpoint_type}/{task_id}"
    start = time.time()
    last_status = None

    while time.time() - start < timeout:
        resp = requests.get(url, headers=api_headers())
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            print(f"Poll error: {data.get('message')}", file=sys.stderr)
            sys.exit(1)

        status = data["data"]["task_status"]
        elapsed = int(time.time() - start)

        if status != last_status:
            print(f"[{elapsed}s] Status: {status}")
            last_status = status

        if status == "succeed":
            return data["data"]["task_result"]
        elif status == "failed":
            msg = data["data"].get("task_status_msg", "Unknown failure")
            print(f"Task failed: {msg}", file=sys.stderr)
            sys.exit(1)

        time.sleep(interval)

    print(f"Timeout after {timeout}s waiting for task {task_id}", file=sys.stderr)
    sys.exit(1)


def download_video(video_url, output_path):
    """Download a video from URL to local file."""
    print(f"Downloading video to {output_path}...")
    resp = requests.get(video_url, stream=True)
    resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB to {output_path}")
    return output_path


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kling AI Video Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text to video
  %(prog)s text2video --prompt "A cat playing piano" --model kling-v2-master --mode pro --duration 10

  # Image to video
  %(prog)s image2video --image photo.jpg --prompt "The scene comes alive" --duration 10

  # Extend a video
  %(prog)s extend --task-id abc123 --prompt "Continue the action"

  # Just check status of an existing task
  %(prog)s status --task-id abc123 --type text2video
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # -- Shared arguments
    def add_common_args(p):
        p.add_argument("--model", default="kling-v2-master",
                        help="Model name (default: kling-v2-master)")
        p.add_argument("--mode", default="pro", choices=["std", "pro"],
                        help="Generation mode (default: pro)")
        p.add_argument("--duration", default="10", choices=["5", "10"],
                        help="Video duration in seconds (default: 10)")
        p.add_argument("--aspect-ratio", default="16:9",
                        choices=["16:9", "9:16", "1:1"],
                        help="Aspect ratio (default: 16:9)")
        p.add_argument("--output", "-o", default="output.mp4",
                        help="Output file path (default: output.mp4)")
        p.add_argument("--negative-prompt", default=None,
                        help="Negative prompt")
        p.add_argument("--sound", default="off", choices=["on", "off"],
                        help="Enable AI sound (kling-v2-6 only)")
        p.add_argument("--cfg-scale", type=float, default=None,
                        help="Prompt adherence 0-1 (v1.x models only)")
        p.add_argument("--timeout", type=int, default=900,
                        help="Max seconds to wait (default: 900)")
        p.add_argument("--poll-interval", type=int, default=5,
                        help="Seconds between status checks (default: 5)")
        p.add_argument("--no-download", action="store_true",
                        help="Don't download, just print the URL")

    # -- text2video
    t2v = subparsers.add_parser("text2video", aliases=["t2v"],
                                 help="Generate video from text prompt")
    t2v.add_argument("--prompt", "-p", required=True, help="Text prompt")
    t2v.add_argument("--camera", default=None,
                      choices=["simple", "down_back", "forward_up",
                               "right_turn_forward", "left_turn_forward"],
                      help="Camera movement preset")
    add_common_args(t2v)

    # -- image2video
    i2v = subparsers.add_parser("image2video", aliases=["i2v"],
                                 help="Generate video from image")
    i2v.add_argument("--image", "-i", required=True,
                      help="Image file path, URL, or base64")
    i2v.add_argument("--image-tail", default=None,
                      help="End frame image (pro mode, v1.5+ only)")
    i2v.add_argument("--prompt", "-p", default=None, help="Text prompt")
    i2v.add_argument("--camera", default=None)
    add_common_args(i2v)

    # -- extend
    ext = subparsers.add_parser("extend", aliases=["ext"],
                                 help="Extend an existing video")
    ext.add_argument("--task-id", required=True, help="Task ID to extend")
    ext.add_argument("--prompt", "-p", default=None, help="Continuation prompt")
    add_common_args(ext)

    # -- status
    st = subparsers.add_parser("status", help="Check task status")
    st.add_argument("--task-id", required=True, help="Task ID to check")
    st.add_argument("--type", required=True,
                     choices=["text2video", "image2video", "video-extend"],
                     help="Endpoint type")
    st.add_argument("--output", "-o", default="output.mp4")
    st.add_argument("--timeout", type=int, default=900)
    st.add_argument("--poll-interval", type=int, default=5)
    st.add_argument("--no-download", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch
    if args.command in ("text2video", "t2v"):
        task_id, etype = create_text2video(args)
    elif args.command in ("image2video", "i2v"):
        task_id, etype = create_image2video(args)
    elif args.command in ("extend", "ext"):
        task_id, etype = create_extend(args)
    elif args.command == "status":
        task_id = args.task_id
        etype = args.type
    else:
        parser.print_help()
        sys.exit(1)

    # Poll
    timeout = getattr(args, "timeout", 900)
    interval = getattr(args, "poll_interval", 5)
    result = poll_task(task_id, etype, timeout=timeout, interval=interval)

    # Download
    videos = result.get("videos", [])
    if not videos:
        print("No videos in result.", file=sys.stderr)
        sys.exit(1)

    video_url = videos[0]["url"]
    video_duration = videos[0].get("duration", "?")
    print(f"\nVideo ready! Duration: {video_duration}s")
    print(f"URL: {video_url}")

    if not getattr(args, "no_download", False):
        output = getattr(args, "output", "output.mp4")
        download_video(video_url, output)
        print(f"\nSaved to: {output}")
    else:
        print(f"\nTask ID: {task_id}")

    # Print task ID for potential extension
    print(f"Task ID (for extend): {task_id}")


if __name__ == "__main__":
    main()
