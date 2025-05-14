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
    parser = argparse.ArgumentParser(description='Process video and extract pose landmarks. Default: process all videos in metadata path in config file')


    parser.add_argument('--video', type=str,
                        help='Path to input video file or directory containing videos (optional if using metadata)')
    parser.add_argument('--output', type=str,
                        help='Output directory for processed landmarks and visualizations')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of detected landmarks')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata CSV file (overrides config value) with optional start/end frames')

    return parser.parse_args(args)


def process_single_video(video_path, output_path, video_metadata, pipeline):
    """Process a single video file"""
    # Get video-specific metadata if available
    video_id = Path(video_path).name
    logging.info(f"Looking for metadata for video ID: {video_id}")

    video_segment = None
    if video_metadata and video_id in video_metadata:
        video_segment = video_metadata[video_id].copy()  # Create a copy to preserve original
        logging.info(f"Found metadata for video {video_id}: {video_segment}")
        if 'start_frame' in video_segment and 'end_frame' in video_segment:
            logging.info(
                f"Processing frames {video_segment['start_frame']} to {video_segment['end_frame']} for {video_id}"
            )
            # Remove from the copy so the entire video is processed
            video_segment.pop('start_frame', None)
            video_segment.pop('end_frame', None)
    else:
        logging.info(f"No metadata found for video {video_id}")

    logging.info(f"Processing video: {video_path}")
    result = pipeline.process_video(video_path, output_path, video_segment)

    if result['success']:
        logging.info(f"Successfully extracted landmarks from {video_path}")
        return True
    else:
        logging.error(f"Failed to extract landmarks from {video_path}: {result.get('error')}")
        return False


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
        args.output = config['paths']['data']['processed_videos']['basic']

    # Determine metadata path (command line takes priority over config)
    metadata_path = None
    if args.metadata:
        # Metadata specified in command line
        metadata_path = args.metadata
        logging.info(f"Using metadata file from command line: {metadata_path}")
    elif 'paths' in config and 'data' in config['paths'] and 'metadata' in config['paths']['data']:
        # Try to get metadata from config
        metadata_dir = config['paths']['data']['metadata'].get('default', '')
        metadata_file = config['paths']['data']['metadata'].get('test_data', '')
        if metadata_dir and metadata_file:
            config_metadata_path = os.path.join(metadata_dir, metadata_file)
            if os.path.exists(config_metadata_path):
                metadata_path = config_metadata_path
                logging.info(f"Using metadata file from config: {config_metadata_path}")
            else:
                logging.warning(f"Metadata file specified in config not found: {config_metadata_path}")

    # Load metadata if a path was determined
    video_metadata = None
    if metadata_path:
        video_metadata = load_video_metadata_file(metadata_path)
        if video_metadata:
            logging.info(f"Loaded metadata for {len(video_metadata)} videos")
            logging.info(f"Loaded metadata for videos: {list(video_metadata.keys())}")
        else:
            logging.warning("No valid metadata loaded, will process entire videos")
    else:
        logging.info("No metadata path available, proceeding without metadata")

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Get the raw video directory from config
    raw_video_dir = config.get('paths', {}).get('data', {}).get('raw', 'data/raw/')

    # Process videos based on command line args or metadata
    if args.video:
        # Process the specific video or directory provided via command line
        if os.path.isfile(args.video):
            # Process single video
            output_path = os.path.join(args.output, Path(args.video).stem)
            process_single_video(args.video, output_path, video_metadata, pipeline)
        elif os.path.isdir(args.video):
            # Process all videos in directory
            from src.core.file_utils import list_files

            video_files = list_files(args.video, extension=('.mp4', '.avi', '.mov'))
            logging.info(f"Found {len(video_files)} video files to process")

            success_count = 0
            for video_path in video_files:
                output_path = os.path.join(args.output, Path(video_path).stem)
                if process_single_video(video_path, output_path, video_metadata, pipeline):
                    success_count += 1

            logging.info(f"Processed {success_count}/{len(video_files)} videos successfully")
        else:
            logging.error(f"Video path does not exist: {args.video}")
            sys.exit(1)
    elif video_metadata:
        # No specific video provided, process all videos in metadata
        logging.info(f"No specific video provided, processing all {len(video_metadata)} videos in metadata")

        success_count = 0
        for video_id in video_metadata.keys():
            # Construct the video path from raw directory and video ID
            video_path = os.path.join(raw_video_dir, video_id)

            # Check if the file exists
            if not os.path.isfile(video_path):
                logging.warning(f"Video file not found: {video_path}")
                continue

            output_path = os.path.join(args.output, Path(video_path).stem)

            logging.info(f"Processing video {success_count + 1}/{len(video_metadata)}: {video_path}")
            if process_single_video(video_path, output_path, video_metadata, pipeline):
                success_count += 1

        logging.info(f"Processed {success_count}/{len(video_metadata)} videos successfully")
    else:
        logging.error("No video specified and no metadata available. Please provide a video path or metadata file.")
        sys.exit(1)


if __name__ == "__main__":
    main()