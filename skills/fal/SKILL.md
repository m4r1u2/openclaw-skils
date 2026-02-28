---
name: fal
description: Generate images and videos using fal.ai models (text-to-image, image-to-video). Use this skill when the user wants to create images from text prompts, convert images to video, or use any fal.ai model endpoint. Supports submit, subscribe, run, stream, batch, realtime WebSocket, file upload, and job management via the fal-client Python library.
metadata: { "openclaw": { "emoji": "🎨", "homepage": "https://fal.ai", "primaryEnv": "FAL_KEY", "requires": { "env": ["FAL_KEY"] } } }
user-invocable: true
---

# fal.ai SKILL — Text-to-Image & Image-to-Video

**OpenClaw SKILL** for generating images and videos using [fal.ai](https://fal.ai) models via the [`fal-client`](https://pypi.org/project/fal-client/) Python library.

## Features

Full coverage of the `fal-client` API:

| Category | Commands | fal-client API |
|---|---|---|
| **Generation** | `submit-t2i`, `submit-i2v` | `submit_async()` |
| **Subscribe** | `subscribe-t2i`, `subscribe-i2v` | `subscribe_async()` — blocks with live queue/progress |
| **Direct Run** | `run-t2i`, `run-i2v` | `run_async()` — single HTTP call, no queue |
| **Streaming** | `stream-t2i`, `stream-i2v` | `stream_async()` — server-sent events |
| **Job Mgmt** | `status`, `result`, `cancel` | `status_async()`, `result_async()`, `cancel_async()` |
| **Upload** | `upload-file`, `upload-image`, `upload-raw` | `upload_file_async()`, `upload_image_async()`, `upload_async()` |
| **Encode** | `encode-file`, `encode-image`, `encode-raw` | `encode_file()`, `encode_image()`, `encode()` |
| **Realtime** | `realtime` | `realtime_async()` — WebSocket send/recv |
| **Batch** | `batch-t2i`, `batch-i2v` | `asyncio.gather()` over `subscribe_async()` |
| **Model Lists** | `list-t2i`, `list-i2v` | Built-in catalog |
| **Model Discovery** | `list-models` | Dynamic fetch from fal.ai REST API |

### Additional flags (all generation commands)

| Flag | Description |
|---|---|
| `--hint` | Routing hint for the fal queue |
| `--priority` | Queue priority (`normal`, `rush`) |
| `--start-timeout` | Server-side start timeout (seconds) |
| `--client-timeout` | Client-side HTTP timeout (seconds) |
| `--webhook-url` | Webhook URL for async result delivery (submit only) |
| `--extra-json` | Arbitrary extra model arguments as JSON |
| `--path` | Sub-path on the model endpoint |

---

## Setup

```bash
pip install fal-client
export FAL_KEY="your-fal-api-key"
```

> Optional: `pip install Pillow` for `upload-image` and `encode-image` commands.

---

## Supported Models

> **`--model` accepts ANY valid fal.ai endpoint** — you are not limited to the models listed here.
> These lists are for discoverability only. fal.ai adds new models frequently.
> Browse all models at https://fal.ai/models

### Text-to-Image

| Model | Endpoint |
|---|---|
| **Nano Banana 2** (Gemini Flash Image) | `fal-ai/nano-banana-2` |
| Nano Banana Pro | `fal-ai/nano-banana-pro` |
| Gemini 3.1 Flash Image Preview | `fal-ai/gemini-3.1-flash-image-preview` |
| **FLUX 2 Flex** | `fal-ai/flux-2-flex` |
| FLUX Dev | `fal-ai/flux/dev` |
| FLUX Schnell | `fal-ai/flux/schnell` |
| FLUX Pro 1.1 Ultra | `fal-ai/flux-pro/v1.1-ultra` |
| FLUX Pro 1.1 | `fal-ai/flux-pro/v1.1` |
| FLUX Pro | `fal-ai/flux-pro` |
| FLUX Pro New | `fal-ai/flux-pro/new` |
| FLUX Realism | `fal-ai/flux-realism` |
| FLUX LoRA | `fal-ai/flux-lora` |
| FLUX LoRA Fill | `fal-ai/flux-lora-fill` |
| FLUX Differential Diffusion | `fal-ai/flux-differential-diffusion` |
| FLUX Krea LoRA (stream) | `fal-ai/flux-krea-lora/stream` |
| **Recraft V4 Pro** | `fal-ai/recraft/v4/pro/text-to-image` |
| Recraft V4 | `fal-ai/recraft/v4/text-to-image` |
| Recraft V4 Pro (vector) | `fal-ai/recraft/v4/pro/text-to-vector` |
| Recraft V4 (vector) | `fal-ai/recraft/v4/text-to-vector` |
| Recraft V3 | `fal-ai/recraft-v3` |
| Recraft V3 | `fal-ai/recraft/v3/text-to-image` |
| Stable Diffusion 3.5 Large | `fal-ai/stable-diffusion-v35-large` |
| Stable Diffusion 3.5 Large Turbo | `fal-ai/stable-diffusion-v35-large/turbo` |
| Stable Diffusion 3.5 Medium | `fal-ai/stable-diffusion-v35-medium` |
| Stable Cascade | `fal-ai/stable-cascade` |
| Seedream 5.0 Lite | `fal-ai/bytedance/seedream/v5/lite/text-to-image` |
| BitDance | `fal-ai/bitdance` |
| Qwen-Image | `fal-ai/qwen-image` |
| Ideogram V2 | `fal-ai/ideogram/v2` |
| Ideogram V2 Turbo | `fal-ai/ideogram/v2/turbo` |
| Bria Fibo Generate | `bria/fibo/generate` |
| ImagineArt 1.5 | `imagineart/imagineart-1.5-preview/text-to-image` |
| AuraFlow | `fal-ai/aura-flow` |
| Kolors | `fal-ai/kolors` |
| Fooocus | `fal-ai/fooocus` |

### Image-to-Video

| Model | Endpoint |
|---|---|
| **Veo 3.1** (Google) | `fal-ai/veo3.1/image-to-video` |
| Veo 3.1 Fast | `fal-ai/veo3.1/fast/image-to-video` |
| Veo 3.1 Reference-to-Video | `fal-ai/veo3.1/reference-to-video` |
| Veo 3.1 First-Last Frame | `fal-ai/veo3.1/first-last-frame-to-video` |
| Veo 3.1 Fast First-Last Frame | `fal-ai/veo3.1/fast/first-last-frame-to-video` |
| **Sora 2** (OpenAI) | `fal-ai/sora-2/image-to-video` |
| Sora 2 Pro | `fal-ai/sora-2/image-to-video/pro` |
| **Kling O3** (Standard) | `fal-ai/kling-video/o3/standard/image-to-video` |
| Kling V3 Pro | `fal-ai/kling-video/v3/pro/image-to-video` |
| Kling V2.5 Turbo Pro | `fal-ai/kling-video/v2.5-turbo/pro/image-to-video` |
| Kling V2.1 Master | `fal-ai/kling-video/v2.1/master/image-to-video` |
| Kling V2.1 Pro | `fal-ai/kling-video/v2.1/pro/image-to-video` |
| Kling V2 Master | `fal-ai/kling-video/v2/master/image-to-video` |
| Kling V1.5 Pro | `fal-ai/kling-video/v1.5/pro/image-to-video` |
| Kling V1 Pro | `fal-ai/kling-video/v1/pro/image-to-video` |
| Kling V1 Standard | `fal-ai/kling-video/v1/standard/image-to-video` |
| MiniMax Video | `fal-ai/minimax-video/image-to-video` |
| MiniMax Hailuo-02 | `fal-ai/minimax/hailuo-02/standard/image-to-video` |
| Runway Gen-3 Turbo | `fal-ai/runway-gen3/turbo/image-to-video` |
| **LTX-2 19B** | `fal-ai/ltx-2-19b/image-to-video` |
| LTX Video 13B Distilled | `fal-ai/ltx-video-13b-distilled/image-to-video` |
| Wan V2.2 | `fal-ai/wan/v2.2-a14b/image-to-video` |
| Wan V2.2 + LoRA | `fal-ai/wan/v2.2-a14b/image-to-video/lora` |
| Wan V2.1 | `fal-ai/wan/v2.1/image-to-video` |
| PixVerse V5 | `fal-ai/pixverse/v5/image-to-video` |
| Cosmos Predict 2.5 (NVIDIA) | `fal-ai/cosmos-predict-2.5/image-to-video` |
| Vidu Q3 Turbo | `fal-ai/vidu/q3/image-to-video/turbo` |
| Lucy (Decart) | `decart/lucy-i2v` |
| Lucy 14B (Decart) | `decart/lucy-14b/image-to-video` |
| Luma Dream Machine | `fal-ai/luma-dream-machine/image-to-video` |
| HunyuanVideo | `fal-ai/hunyuan-video/image-to-video` |
| CogVideoX-5B | `fal-ai/cogvideox-5b/image-to-video` |
| Stable Video | `fal-ai/stable-video` |
| HeyGen Avatar 4 | `fal-ai/heygen/avatar4/image-to-video` |
| Creatify Aurora | `fal-ai/creatify/aurora` |
| VEED Fabric 1.0 | `veed/fabric-1.0` |
| Omnihuman V1.5 | `fal-ai/bytedance/omnihuman/v1.5` |
| MultiTalk Avatar | `fal-ai/ai-avatar/single-text` |

---

## Usage Examples

### Text-to-Image

```bash
# Subscribe — wait for result with live progress
python skill_fal.py subscribe-t2i \
  --model fal-ai/flux/dev \
  --prompt "A cyberpunk cityscape at sunset, neon lights reflecting on wet streets"

# Submit — fire-and-forget, returns request_id
python skill_fal.py submit-t2i \
  --model fal-ai/flux/schnell \
  --prompt "A watercolor painting of a forest" \
  --width 1024 --height 768 --num-images 2

# Direct run — no queue, single HTTP call
python skill_fal.py run-t2i \
  --model fal-ai/flux/schnell \
  --prompt "A minimalist logo design"

# Stream — server-sent events
python skill_fal.py stream-t2i \
  --model fal-ai/flux/dev \
  --prompt "Abstract geometric art"

# With all options
python skill_fal.py subscribe-t2i \
  --model fal-ai/flux/dev \
  --prompt "Photorealistic mountain landscape" \
  --negative-prompt "blurry, low quality" \
  --width 1024 --height 1024 \
  --num-images 1 \
  --seed 42 \
  --guidance-scale 7.5 \
  --num-inference-steps 30 \
  --extra-json '{"scheduler": "euler"}' \
  --priority rush \
  --start-timeout 300
```

### Image-to-Video

```bash
# Subscribe with URL
python skill_fal.py subscribe-i2v \
  --model fal-ai/runway-gen3/turbo/image-to-video \
  --image-url "https://example.com/cat.png" \
  --prompt "The cat slowly turns its head"

# Submit with local file (auto-uploaded to fal CDN)
python skill_fal.py submit-i2v \
  --model fal-ai/kling-video/v2/master/image-to-video \
  --image-url ./my-photo.jpg \
  --duration 5.0

# With webhook for async delivery
python skill_fal.py submit-i2v \
  --model fal-ai/minimax-video/image-to-video \
  --image-url "https://example.com/scene.png" \
  --webhook-url "https://myapp.com/api/fal-callback"
```

### Job Management

```bash
# Check status
python skill_fal.py status \
  --model fal-ai/flux/dev \
  --request-id abc123-def456

# Fetch result
python skill_fal.py result \
  --model fal-ai/flux/dev \
  --request-id abc123-def456

# Cancel
python skill_fal.py cancel \
  --model fal-ai/flux/dev \
  --request-id abc123-def456
```

### Upload & Encode

```bash
# Upload local file → CDN URL
python skill_fal.py upload-file --file ./photo.jpg
# → https://fal.media/files/...

# Upload PIL image → CDN URL
python skill_fal.py upload-image --file ./photo.png --format png

# Upload raw bytes
python skill_fal.py upload-raw --file ./data.bin --content-type application/octet-stream

# Encode file to base64 data URI (no network)
python skill_fal.py encode-file --file ./photo.jpg
# → data:image/jpeg;base64,/9j/4AAQ...

# Encode image via PIL
python skill_fal.py encode-image --file ./photo.png --format png
```

### Realtime (WebSocket)

```bash
python skill_fal.py realtime \
  --model fal-ai/flux/schnell \
  --arguments-json '{"prompt": "A glowing orb", "num_images": 1}' \
  --rounds 3
```

### Batch Processing

```bash
# Multiple prompts in parallel
python skill_fal.py batch-t2i \
  --model fal-ai/flux/schnell \
  --prompts "A red car" "A blue boat" "A green plane" \
  --width 512 --height 512

# Multiple images to video in parallel
python skill_fal.py batch-i2v \
  --model fal-ai/runway-gen3/turbo/image-to-video \
  --image-urls "https://example.com/a.png" "https://example.com/b.png"
```

### Dynamic Model Discovery

```bash
# List all available categories and counts
python skill_fal.py list-models --categories

# List all text-to-image models (live from fal.ai API)
python skill_fal.py list-models --category text-to-image

# List image-to-video models
python skill_fal.py list-models --category image-to-video

# Search for a specific model by name
python skill_fal.py list-models --search "nano-banana"

# Include deprecated models
python skill_fal.py list-models --category text-to-image --include-deprecated

# Full JSON output (all model metadata)
python skill_fal.py list-models --category text-to-video --json

# All models across all categories
python skill_fal.py list-models
```

---

## Output

- **JSON results** are printed to **stdout** (pipe-friendly).
- **Progress / status logs** are printed to **stderr**.
- Submit commands return `{"request_id": "...", "model": "..."}`.
- Subscribe / run commands return the full model response JSON.

```bash
# Pipe image URL to clipboard
python skill_fal.py subscribe-t2i \
  --model fal-ai/flux/schnell \
  --prompt "A sunset" 2>/dev/null | jq -r '.images[0].url' | pbcopy
```

---

## Architecture

```
skill_fal.py
├── Dynamic model discovery (fetch_models — fal.ai REST API)
├── Model catalogs (TEXT_TO_IMAGE_MODELS, IMAGE_TO_VIDEO_MODELS)
├── Argument builders (_build_t2i_arguments, _build_i2v_arguments)
├── Core async ops (submit_job, subscribe_job, run_job, stream_job)
├── Job management (check_status, fetch_result, cancel_job)
├── Upload helpers (upload_file, upload_image, upload_raw)
├── Encode helpers (encode_file, encode_image, encode_raw)
├── Realtime (realtime_session)
├── Batch helpers (batch_text_to_image, batch_image_to_video)
├── Auto-upload (_maybe_upload — local paths → CDN)
└── CLI (argparse with 22 subcommands)
```

All generation commands use the full async `fal_client` API (`submit_async`, `subscribe_async`, `run_async`, `stream_async`) inside an `asyncio.run()` event loop.

---

## fal-client API Coverage

| fal-client function | Covered | Command(s) |
|---|---|---|
| `submit` / `submit_async` | ✅ | `submit-t2i`, `submit-i2v` |
| `subscribe` / `subscribe_async` | ✅ | `subscribe-t2i`, `subscribe-i2v`, `batch-*` |
| `run` / `run_async` | ✅ | `run-t2i`, `run-i2v` |
| `stream` / `stream_async` | ✅ | `stream-t2i`, `stream-i2v` |
| `status` / `status_async` | ✅ | `status` |
| `result` / `result_async` | ✅ | `result` |
| `cancel` / `cancel_async` | ✅ | `cancel` |
| `upload` / `upload_async` | ✅ | `upload-raw` |
| `upload_file` / `upload_file_async` | ✅ | `upload-file` |
| `upload_image` / `upload_image_async` | ✅ | `upload-image` |
| `encode` | ✅ | `encode-raw` |
| `encode_file` | ✅ | `encode-file` |
| `encode_image` | ✅ | `encode-image` |
| `realtime` / `realtime_async` | ✅ | `realtime` |
| `on_queue_update` callback | ✅ | All subscribe commands |
| `webhook_url` | ✅ | `submit-t2i`, `submit-i2v` |
| `priority` | ✅ | All generation commands |
| `hint` | ✅ | All generation commands |
| `start_timeout` | ✅ | All generation commands |
| `client_timeout` | ✅ | Subscribe / run / stream |
| `path` (sub-endpoint) | ✅ | All generation commands |
| `SyncClient` / `AsyncClient` | ✅ | Module-level functions used |
| `SyncRequestHandle` / `AsyncRequestHandle` | ✅ | Via `submit_async()` |

> **`ws_connect` / `ws_connect_async`**: Not directly exposed as a CLI command. Use `realtime` for the higher-level WebSocket API. Raw WebSocket access is available programmatically via `fal_client.ws_connect_async()`.
