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

# Configure root logger
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set all loggers to INFO level
logging.getLogger().setLevel(logging.INFO)

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_video_metadata_file
from src.core.pipeline import Pipeline


def parse_args(args=None):
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

    return parser.parse_args(args)


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
        video_metadata = load_video_metadata_file(args.metadata)
        if video_metadata:
            logging.info(f"Loaded metadata for {len(video_metadata)} videos")
        else:
            logging.warning("No valid metadata loaded, will process entire videos")

    # After loading video_metadata
    logging.info(f"Loaded metadata for videos: {list(video_metadata.keys()) if video_metadata else 'None'}")
    if video_id in video_metadata:
        logging.info(f"Found metadata for video {video_id}: {video_metadata[video_id]}")
    else:
        logging.info(f"No metadata found for video {video_id}")


    # Initialize pipeline
    pipeline = Pipeline(config)

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
                    f"Processing frames {video_segment['start_frame']} to {video_segment['end_frame']} for {video_id}"
                )

        logging.info(f"Processing video: {video_path}")
        result = pipeline.process_video(video_path, output_path, video_segment)

        if result['success']:
            logging.info(f"Successfully extracted landmarks from {video_path}")
        else:
            logging.error(f"Failed to extract landmarks from {video_path}: {result.get('error')}")

    elif os.path.isdir(args.video):
        # Process all videos in directory
        from src.core.file_utils import list_files

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
                        f"Processing frames {video_segment['start_frame']} to {video_segment['end_frame']} for {video_id}"
                    )

            logging.info(f"Processing video: {video_path}")
            result = pipeline.process_video(video_path, output_path, video_segment)

            if result['success']:
                success_count += 1
                logging.info(f"Successfully extracted landmarks from {video_path}")
            else:
                logging.error(f"Failed to extract landmarks from {video_path}: {result.get('error')}")

        logging.info(f"Processed {success_count}/{len(video_files)} videos successfully")
    else:
        logging.error(f"Video path does not exist: {args.video}")
        sys.exit(1)


if __name__ == "__main__":
    main()