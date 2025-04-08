# scripts/process_video.py
# !/usr/bin/env python3
"""
Script to process videos and extract pose landmarks using the configured pose estimator.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigLoader
from pose_estimators.mediapipe_estimator import MediaPipePoseEstimator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process video and extract pose landmarks')

    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file or directory containing videos')
    parser.add_argument('--output', type=str,
                        help='Output directory for processed landmarks and visualizations')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of detected landmarks')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata CSV file with optional start/end frames')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    if not config:
        logging.error(f"Failed to load configuration: {args.config}")
        sys.exit(1)

    # Set default output directory if not specified
    if not args.output:
        args.output = config['paths']['data']['landmarks']

    # Load metadata if provided
    video_metadata = None
    if args.metadata:
        from core.file_utils import load_video_metadata_file
        video_metadata = load_video_metadata_file(args.metadata)
        if video_metadata:
            logging.info(f"Loaded metadata for {len(video_metadata)} videos")
        else:
            logging.warning("No valid metadata loaded, will process entire videos")

    # Initialize pose estimator
    pose_estimator = MediaPipePoseEstimator(config['pose_estimation'])

    # Process single video or directory of videos
    if os.path.isfile(args.video):
        # Process single video
        video_path = args.video
        output_path = os.path.join(args.output, Path(video_path).stem)

        # Get video-specific metadata if available
        video_id = Path(video_path).stem
        video_segment = None
        if video_metadata and video_id in video_metadata:
            video_segment = video_metadata[video_id]
            if 'start_frame' in video_segment and 'end_frame' in video_segment:
                logging.info(
                    f"Processing frames {video_segment['start_frame']} to {video_segment['end_frame']} for {video_id}")

        logging.info(f"Processing video: {video_path}")
        landmarks = pose_estimator.process_video(video_path, output_path, video_segment)

        if landmarks is not None:
            logging.info(f"Successfully extracted landmarks from {video_path}")
        else:
            logging.error(f"Failed to extract landmarks from {video_path}")

    elif os.path.isdir(args.video):
        # Process all videos in directory
        from core.file_utils import list_files

        video_files = list_files(args.video, extension=('.mp4', '.avi', '.mov'))
        logging.info(f"Found {len(video_files)} video files to process")

        success_count = 0
        for video_path in video_files:
            output_path = os.path.join(args.output, Path(video_path).stem)

            # Get video-specific metadata if available
            video_id = Path(video_path).stem
            video_segment = None
            if video_metadata and video_id in video_metadata:
                video_segment = video_metadata[video_id]
                if 'start_frame' in video_segment and 'end_frame' in video_segment:
                    logging.info(
                        f"Processing frames {video_segment['start_frame']} to {video_segment['end_frame']} for {video_id}")

            logging.info(f"Processing video: {video_path}")
            landmarks = pose_estimator.process_video(video_path, output_path, video_segment)

            if landmarks is not None:
                success_count += 1
                logging.info(f"Successfully extracted landmarks from {video_path}")
            else:
                logging.error(f"Failed to extract landmarks from {video_path}")

        logging.info(f"Processed {success_count}/{len(video_files)} videos successfully")
    else:
        logging.error(f"Video path does not exist: {args.video}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()