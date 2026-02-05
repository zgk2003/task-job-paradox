#!/usr/bin/env python3
"""
Generate video presentation from slides and narration.

Requirements:
    pip install gtts moviepy pillow

Usage:
    python create_video.py
"""

import os
from pathlib import Path

# Check dependencies
try:
    from gtts import gTTS
except ImportError:
    print("Please install gTTS: pip install gtts")
    exit(1)

try:
    from moviepy.editor import (
        ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
    )
except ImportError:
    print("Please install moviepy: pip install moviepy")
    exit(1)

PRESENTATION_DIR = Path(__file__).parent
SLIDES_DIR = PRESENTATION_DIR / 'slides'
AUDIO_DIR = PRESENTATION_DIR / 'audio'
OUTPUT_FILE = PRESENTATION_DIR / 'task_job_paradox_presentation.mp4'

AUDIO_DIR.mkdir(exist_ok=True)

# Narration text for each slide
NARRATIONS = {
    'slide_01_title.png': """
Welcome to this presentation on the Task-Job Paradox.
This research examines a puzzling phenomenon: while AI coding tools make individual tasks faster,
we don't see a corresponding increase in total developer output.
We analyzed five years of GitHub data, using the ChatGPT launch as a natural experiment.
""",

    'slide_02_puzzle.png': """
What's the paradox? On one side, AI tools make developers 30 to 50 percent faster on individual tasks.
Code completion, bug fixes, writing tests - all get done more quickly.
But on the other side, total output metrics like pull requests merged stay roughly the same.
This raises a fundamental question: where does the saved time go?
""",

    'slide_03_method.png': """
We collected 53 months of GitHub Archive data - over 11 million observations.
We tracked velocity, complexity, and throughput metrics.
For causal identification, we used Difference-in-Differences, comparing Python and JavaScript
against control languages like Fortran and COBOL that are unaffected by AI tools.
We also used Interrupted Time Series analysis around the ChatGPT launch.
""",

    'slide_04_velocity.png': """
Our first finding: velocity improved significantly.
The Difference-in-Differences analysis found a reduction of 4.62 hours in PR lead time
for high-exposure languages compared to the control group.
This confirms that AI tools genuinely accelerate task completion.
""",

    'slide_05_scope.png': """
Here's where it gets interesting. Our second finding shows scope expansion.
PR sizes increased by over 10 lines in high-exposure languages, with a p-value less than 0.0001.
Developers aren't using saved time to produce more pull requests.
Instead, they're tackling larger changes. Speed gains get absorbed into increased ambition.
""",

    'slide_06_did.png': """
These charts show the Difference-in-Differences results.
High-exposure languages got faster and show larger PRs,
while the control group remains unchanged.
This confirms we're seeing a causal effect of AI tools.
""",

    'slide_07_mechanism.png': """
The mechanism is straightforward. AI tools make tasks faster.
But instead of producing more output, developers expand scope.
Time saved is absorbed by increased ambition.
It's like highway expansion - adding lanes doesn't reduce commute times
because more people start driving.
""",

    'slide_08_conclusion.png': """
Our main takeaway: AI changes how we work, not how much we produce.
Productivity metrics need to account for scope changes.
AI tools enable ambition more than pure efficiency.
This research was conducted with AI assistance - we experienced the paradox firsthand.
Thank you for your attention.
""",
}


def generate_audio():
    """Generate audio files for each slide using gTTS."""
    print("Generating audio narration...")

    audio_files = []
    for slide_name, text in NARRATIONS.items():
        audio_path = AUDIO_DIR / slide_name.replace('.png', '.mp3')

        if audio_path.exists():
            print(f"  Skipping {slide_name} (audio exists)")
            audio_files.append(audio_path)
            continue

        print(f"  Generating audio for {slide_name}...")
        tts = gTTS(text=text.strip(), lang='en', slow=False)
        tts.save(str(audio_path))
        audio_files.append(audio_path)

    return audio_files


def create_video():
    """Combine slides and audio into a video."""
    print("\nCreating video...")

    clips = []

    for slide_name in NARRATIONS.keys():
        slide_path = SLIDES_DIR / slide_name
        audio_path = AUDIO_DIR / slide_name.replace('.png', '.mp3')

        if not slide_path.exists():
            print(f"  Warning: {slide_path} not found, skipping")
            continue

        if not audio_path.exists():
            print(f"  Warning: {audio_path} not found, skipping")
            continue

        print(f"  Processing {slide_name}...")

        # Load audio to get duration
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration + 1  # Add 1 second buffer

        # Create image clip with same duration as audio
        img_clip = ImageClip(str(slide_path)).set_duration(duration)
        img_clip = img_clip.set_audio(audio)

        clips.append(img_clip)

    if not clips:
        print("No clips to combine!")
        return

    # Concatenate all clips
    print("\nConcatenating clips...")
    final = concatenate_videoclips(clips, method="compose")

    # Write output
    print(f"\nWriting video to {OUTPUT_FILE}...")
    final.write_videofile(
        str(OUTPUT_FILE),
        fps=24,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    print(f"\nVideo created: {OUTPUT_FILE}")
    print(f"Duration: {final.duration:.1f} seconds")


def main():
    print("=" * 50)
    print("TASK-JOB PARADOX VIDEO GENERATOR")
    print("=" * 50)

    # Check if slides exist
    if not SLIDES_DIR.exists() or not list(SLIDES_DIR.glob('*.png')):
        print("Slides not found! Run create_slides.py first.")
        return

    # Generate audio
    generate_audio()

    # Create video
    create_video()


if __name__ == '__main__':
    main()
