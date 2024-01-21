# Roulette Video Processor

## Overview
This project is a proof of concept designed to explore the capabilities of computer vision in the context of object tracking. Using OpenCV and YOLO (You Only Look Once). The implementation includes `RouletteWheelTracker` and `RouletteBallTracker` for tracking the roulette wheel and ball, respectively.

## Prerequisites
- Python 3.10
- **Important:** PyTorch must be installed before proceeding with the installation of other dependencies. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for detailed installation instructions.

## Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/cbaezp/roulette
cd roulette
```

After installing PyTorch, install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To run the video processing script, navigate to the repository's root directory and execute:

```bash
python roulette.py
```

### Parameters
- `VIDEO_PATH`: Path to the input video file (e.g., `videos/roulette_test.mp4`).
- `OUTPUT_PATH`: Path for the output video file (e.g., `output_video.mp4`).
- Both, `RouletteWheelTracker` and `RouletteBallTracker` have additional parameters that could be updated based on the video, angle, etc.


### Customization
You can customize the behavior of the trackers by modifying their initialization parameters in `roulette.py`.

## Components
### VideoProcessor
Manages the video processing pipeline, handling frame reading and writing operations.

### RouletteWheelTracker
Detects and tracks the roulette wheel in video frames.

### RouletteBallTracker
Utilizes YOLO, powered by PyTorch, for the precise tracking of the ball's position and trajectory.

## Proof of Concept
This project demonstrates the practical application of computer vision techniques, showcasing object tracking under dynamic conditions.


## YouTube
[![Check out the youtube video]](https://youtu.be/bpy933SQ6Q0)


