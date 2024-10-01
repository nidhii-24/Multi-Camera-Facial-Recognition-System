# Multi-Camera Facial Recognition System

This project implements a multi-camera facial recognition system for person detection and tracking. It includes modules for face detection, recognition, and reidentification across multiple cameras.

## Features

- Face detection and capture of new individuals from multiple cameras
- Face recognition and tracking across multiple cameras
- Real-time face detection and recognition using multiple webcams
- Database storage for known and unknown faces
- Automatic creation of new person entries in the database
- Common database for all cameras

## Requirements

- At least two cameras (webcams or IP cameras) connected to your system
- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
2. Install the required dependencies:
## Usage

1. Ensure that at least two cameras are connected to your system.

2. Run the main script:
3. The system will open windows for each camera feed, showing the detected and recognized faces.

4. Press 'c' in any camera window to capture a new person (30 frames will be captured from that camera).

5. Press 'q' in any camera window to quit the application.
6.You can easily extend this to more than two cameras by modifying the range in the main() function: for camera_id in range(num_cameras):  # Replace 2 with the number of cameras you want to use

## Project Structure

- `face_detection/`: Module for face detection and new person capture
- `face_recognition/`: Module for face recognition and tracking
- `database/`: Storage for known and unknown faces (shared across all cameras)
- `main.py`: Main script to run the multi-camera facial recognition system
- `requirements.txt`: List of required Python packages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

