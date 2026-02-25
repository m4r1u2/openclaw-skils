---
name: kling-video
description: Generate AI videos using the Kling AI API (klingai.com). Use this skill when the user wants to create videos from text prompts or images, generate YouTube content, create animated shorts, or produce video clips. Supports text-to-video, image-to-video, and video extension. Can produce YouTube-ready 16:9 videos with sound.
metadata: { "openclaw": { "emoji": "🎬", "homepage": "https://klingai.com", "primaryEnv": "KLING_ACCESS_KEY", "requires": { "env": ["KLING_ACCESS_KEY", "KLING_SECRET_KEY"] } } }
user-invocable: true
---

# Kling AI Video Generation Skill

Generate AI videos using the Kling AI API. This skill handles the full workflow from prompt crafting to downloading finished videos ready for YouTube upload.

## Environment Variables Required

- `KLING_ACCESS_KEY` — Your Kling AI access key (get from https://klingai.com/global/dev/model-api)
- `KLING_SECRET_KEY` — Your Kling AI secret key

## Quick Start

When the user asks to generate a video, follow these steps:

1. **Craft an optimized prompt** (see Prompt Engineering below)
2. **Generate the video** using the script at `{baseDir}/scripts/kling.py`
3. **Poll until complete** and download the result
4. **Report** the output file path to the user

## Usage

### Text-to-Video

```bash
python3 {baseDir}/scripts/kling.py text2video \
  --prompt "Minecraft Steve versus Mickey Mouse in tekken style fighting, funny slapstick comedy, anime style, dynamic action" \
  --model kling-v2-master \
  --mode pro \
  --duration 10 \
  --aspect-ratio 16:9 \
  --output ./output_video.mp4
```

### Image-to-Video

```bash
python3 {baseDir}/scripts/kling.py image2video \
  --image ./start_frame.png \
  --prompt "The character starts running with exaggerated anime speed lines" \
  --model kling-v2-master \
  --mode pro \
  --duration 10 \
  --aspect-ratio 16:9 \
  --output ./output_video.mp4
```

### Extend a Video

```bash
python3 {baseDir}/scripts/kling.py extend \
  --task-id <previous_task_id> \
  --prompt "Continue the action with an epic finishing move" \
  --output ./extended_video.mp4
```

### Available Models

| Model | Best For | Resolution | Sound |
|-------|----------|------------|-------|
| `kling-v2-master` | Highest quality, best motion | 720p (24fps) | No |
| `kling-v2-6` | Quality + sound generation | 720p | Yes |
| `kling-v1-6` | Good quality, 1080p in pro | 720p/1080p (30fps) | No |
| `kling-v2-5-turbo` | Fast generation | 720p | No |

### Recommended Settings for YouTube

- **Model**: `kling-v2-master` (best quality) or `kling-v2-6` (if sound needed)
- **Mode**: `pro` (higher quality, takes longer)
- **Duration**: `10` (10 seconds per clip)
- **Aspect ratio**: `16:9` (standard YouTube)

## Prompt Engineering Guide

Kling AI responds best to structured, descriptive prompts. Follow these rules:

### Prompt Structure

```
[Subject/Characters] + [Action/Motion] + [Style/Aesthetic] + [Camera/Cinematography] + [Mood/Atmosphere]
```

### Rules for Great Prompts

1. **Be specific about characters**: Describe them visually — "blocky pixelated Minecraft Steve with diamond armor" not just "Minecraft Steve"
2. **Describe motion explicitly**: "throwing rapid punches", "dodging with a backflip", "landing a spinning kick"
3. **Specify art style clearly**: "cel-shaded anime style", "Pixar 3D animation", "retro pixel art", "watercolor painting"
4. **Include camera direction**: "dynamic low angle tracking shot", "close-up on face", "wide establishing shot"
5. **Add atmosphere/mood**: "dramatic spotlights", "dust particles in the air", "speed lines and impact frames"
6. **Max 2500 characters** per prompt

### Example Prompts for Fighting/Comedy Content

**Epic Fight Scene:**
```
Blocky pixelated Minecraft Steve in diamond armor versus classic cartoon Mickey Mouse in a dramatic fighting game arena. Steve throws a rapid combo of blocky punches while Mickey dodges with exaggerated cartoon flexibility. Anime style with speed lines, impact frames, and dramatic lighting. Dynamic camera angles switching between close-ups and wide shots. Tekken-style fighting game HUD visible. Funny slapstick comedy with over-the-top reactions. Vibrant colors, particle effects on each hit.
```

**Comedy Slapstick:**
```
Mickey Mouse attempts a spinning kick but slips on a banana peel, sliding across a colorful fighting arena floor. Minecraft Steve watches confused then gets hit by the sliding Mickey. Exaggerated anime reaction faces, sweat drops, and exclamation marks appear. Cel-shaded anime art style with bold outlines. Camera follows the action with comedic timing. Dramatic pause before impact then explosion of cartoon stars and effects.
```

**Dramatic Finish:**
```
Epic final round in a neon-lit fighting arena. Minecraft Steve charges a glowing diamond sword ultimate attack. Mickey Mouse counters with a giant cartoon mallet. Both attacks collide creating a massive energy explosion. Anime style close-up of both characters' determined faces before impact. Dynamic camera rotation around the clash. Dramatic lighting with lens flares. Speed lines radiate from the center of impact.
```

### Negative Prompt Suggestions

For cleaner output, use negative prompts:
```
blurry, distorted faces, extra limbs, deformed hands, low quality, watermark, text overlay, static image, no motion
```

## YouTube Video Production Workflow

For creating a full YouTube video from multiple Kling clips:

### Step 1: Plan the Scene Sequence
Break your video idea into 5-10 second clips with clear actions per scene.

### Step 2: Generate Each Clip
Generate each scene individually, maintaining consistent style keywords across all prompts.

### Step 3: Use Video Extension
For longer scenes, use the `extend` command to continue a clip for another 5-10 seconds.

### Step 4: Download All Clips
All clips are saved as MP4 files. They can be concatenated with ffmpeg:

```bash
# Create a file list
for f in clip_*.mp4; do echo "file '$f'" >> filelist.txt; done

# Concatenate
ffmpeg -f concat -safe 0 -i filelist.txt -c copy final_video.mp4
```

### Step 5: Add Audio (if not using kling-v2-6 sound)
```bash
# Add background music
ffmpeg -i final_video.mp4 -i background_music.mp3 -c:v copy -c:a aac -shortest youtube_ready.mp4
```

## Tips

- **Batch generation**: Generate multiple variations of each scene and pick the best one
- **Consistency**: Keep the same style keywords (e.g., "cel-shaded anime style") across all prompts for a cohesive look
- **Video extension**: Use extend to create longer sequences from the best clips
- **Sound**: Use `kling-v2-6` with `--sound on` if you want AI-generated sound effects
- **Cost control**: Use `std` mode and `5` second duration for test runs, switch to `pro` and `10` for final renders
