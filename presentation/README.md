# Task-Job Paradox Presentation

5-minute video presentation explaining the research and results.

## Contents

- `slides/` - PNG images for each slide (8 slides)
- `narration_script.md` - Full narration text with timing
- `create_slides.py` - Python script to regenerate slides
- `create_video.py` - Python script to generate video with audio

## Quick Start

### 1. Slides Only (Already Generated)

View the slides in `slides/` directory:
- slide_01_title.png
- slide_02_puzzle.png
- slide_03_method.png
- slide_04_velocity.png
- slide_05_scope.png
- slide_06_did.png
- slide_07_mechanism.png
- slide_08_conclusion.png

### 2. Generate Video with Audio

Install dependencies:
```bash
pip install gtts moviepy pillow
```

Run the video generator:
```bash
python create_video.py
```

This will:
1. Generate MP3 audio files for each slide using Google Text-to-Speech
2. Combine slides and audio into `task_job_paradox_presentation.mp4`

### 3. Alternative: Use PowerPoint/Keynote

Import the PNG slides into your preferred presentation software and record narration manually using the script in `narration_script.md`.

## Slide Overview

| Slide | Title | Duration |
|-------|-------|----------|
| 1 | The Task-Job Paradox | 35s |
| 2 | The Puzzle | 40s |
| 3 | Our Approach | 45s |
| 4 | Finding 1: Tasks Got Faster | 40s |
| 5 | Finding 2: PRs Got Bigger | 40s |
| 6 | The DiD Evidence | 35s |
| 7 | The Mechanism | 40s |
| 8 | Conclusion | 35s |

**Total: ~5 minutes**

## Regenerating Slides

If you need to modify and regenerate slides:
```bash
python create_slides.py
```
