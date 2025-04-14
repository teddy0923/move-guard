# scripts/generate_visualizations.py
# !/usr/bin/env python3
"""
Script to generate visualizations of movement analysis with optional angle overlays.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_landmarks, ensure_directory_exists
from src.core.pipeline import Pipeline


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate movement analysis visualizations')

    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--landmarks', type=str, required=True,
                        help='Path to landmarks file (.npy)')
    parser.add_argument('--output', type=str,
                        help='Output directory for visualization videos')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata CSV file with optional start/end frames')
    parser.add_argument('--angles', action='store_true',
                        help='Visualize joint angles')
    parser.add_argument('--features', type=str,
                        help='Path to extracted features file to visualize metrics')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')
    parser.add_argument('--movement', type=str, default='squat',
                        help='Movement type (e.g., squat, ybt)')
    parser.add_argument('--fps', type=int,
                        help='Output video FPS (defaults to original video FPS)')

    return parser.parse_args(args)


def draw_landmarks(frame, landmarks, connections, color=(0, 255, 0), thickness=2):
    """Draw landmarks and connections on frame"""
    h, w = frame.shape[:2]

    # Draw landmarks
    for landmark in landmarks:
        # Convert normalized coordinates to pixel values
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(frame, (x, y), 5, color, -1)

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
        end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
        cv2.line(frame, start_point, end_point, color, thickness)

    return frame


def calculate_angle(landmarks, p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array([landmarks[p1][0], landmarks[p1][1]])
    b = np.array([landmarks[p2][0], landmarks[p2][1]])
    c = np.array([landmarks[p3][0], landmarks[p3][1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def draw_angle(frame, landmarks, p1, p2, p3, color=(0, 0, 255), thickness=2):
    """Draw angle visualization on frame"""
    h, w = frame.shape[:2]

    # Convert normalized coordinates to pixel values
    p1_coords = (int(landmarks[p1][0] * w), int(landmarks[p1][1] * h))
    p2_coords = (int(landmarks[p2][0] * w), int(landmarks[p2][1] * h))
    p3_coords = (int(landmarks[p3][0] * w), int(landmarks[p3][1] * h))

    # Draw angle lines
    cv2.line(frame, p2_coords, p1_coords, color, thickness)
    cv2.line(frame, p2_coords, p3_coords, color, thickness)

    # Calculate angle
    angle = calculate_angle(landmarks, p1, p2, p3)

    # Draw angle value
    cv2.putText(frame, f"{angle:.1f}°", p2_coords, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, thickness, cv2.LINE_AA)

    return frame


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_loader = ConfigLoader()
    base_config = config_loader.load_config(args.config)

    if not base_config:
        logging.error(f"Failed to load base configuration: {args.config}")
        sys.exit(1)

    # Load movement-specific configuration
    movement_config = config_loader.load_config(args.movement)
    if movement_config:
        config = config_loader.merge_configs(args.config, args.movement)
    else:
        config = base_config

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Set default output directory if not specified
    if not args.output:
        args.output = base_config['paths']['data']['processed_videos']['annotated']

    # Ensure output directory exists
    ensure_directory_exists(args.output)

    # Load landmarks
    landmarks_data = load_landmarks(args.landmarks)
    if landmarks_data is None:
        logging.error(f"Failed to load landmarks from {args.landmarks}")
        sys.exit(1)

    # Load video
    video_path = args.video
    video_id = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        sys.exit(1)

    # Get video properties
    fps = args.fps if args.fps else int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine start and end frames
    start_frame = 0
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Load metadata if provided
    if args.metadata:
        from core.file_utils import load_video_metadata_file
        video_metadata = load_video_metadata_file(args.metadata)

        if video_metadata and video_id in video_metadata:
            metadata_entry = video_metadata[video_id]
            if 'start_frame' in metadata_entry and 'end_frame' in metadata_entry:
                start_frame = metadata_entry['start_frame']
                end_frame = metadata_entry['end_frame']
                logging.info(f"Using frames {start_frame} to {end_frame} from metadata")

    # Validate landmark-video correspondence
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_landmarks = len(landmarks_data)

    if total_landmarks < (end_frame - start_frame + 1):
        logging.warning(f"Warning: Video has {end_frame - start_frame + 1} frames in the specified range, "
                        f"but landmarks data only has {total_landmarks} frames")
        end_frame = start_frame + total_landmarks - 1
        logging.warning(f"Adjusting end_frame to {end_frame}")

    # Additional validation can also verify metadata if available
    if args.metadata:
        # Check if video ID in landmarks filename matches the video ID from the video file
        landmarks_id = Path(args.landmarks).stem
        if landmarks_id != video_id and not landmarks_id.startswith(video_id):
            logging.warning(
                f"Warning: Possible mismatch between video ID ({video_id}) and landmarks ID ({landmarks_id})"
            )

    # Setup video writer
    output_path = os.path.join(args.output, f"{video_id}_visualized.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get pose estimator from pipeline for connection information
    pose_estimator = pipeline.get_component('pose_estimator')

    # Define body connections for visualization (customize based on your pose estimator)
    # This is a simplification and should be adjusted for your specific landmark format
    body_connections = [
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15),
        # Right arm
        (12, 14), (14, 16),
        # Left leg
        (23, 25), (25, 27), (27, 31), (31, 29),
        # Right leg
        (24, 26), (26, 28), (28, 32), (30, 32)
    ]

    # Define angles to visualize if enabled
    angles_to_visualize = []
    if args.angles:
        if args.movement.lower() == 'squat':
            # Hip angle (shoulder-hip-knee)
            angles_to_visualize.append((11, 23, 25))  # Left side
            angles_to_visualize.append((12, 24, 26))  # Right side

            # Knee angle (hip-knee-ankle)
            angles_to_visualize.append((23, 25, 27))  # Left side
            angles_to_visualize.append((24, 26, 28))  # Right side

            # Ankle angle (knee-ankle-foot)
            angles_to_visualize.append((25, 27, 31))  # Left side
            angles_to_visualize.append((26, 28, 32))  # Right side

    # Process video
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames before start_frame
        if frame_idx < start_frame:
            frame_idx += 1
            continue

        # Stop after end_frame
        if frame_idx > end_frame:
            break

        # Get landmarks for current frame
        if frame_idx - start_frame < len(landmarks_data):
            frame_landmarks = landmarks_data[frame_idx - start_frame]

            # Draw landmarks and connections
            frame = draw_landmarks(frame, frame_landmarks, body_connections)

            # Draw angles if enabled
            if args.angles:
                for p1, p2, p3 in angles_to_visualize:
                    frame = draw_angle(frame, frame_landmarks, p1, p2, p3)

            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Write frame to output video
        out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    logging.info(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()