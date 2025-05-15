# scripts/extract_features.py
# !/usr/bin/env python3
"""
Script to extract features from pose landmarks for movement analysis.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_landmarks, save_features, list_files, ensure_directory_exists, load_metadata
from src.core.pipeline import Pipeline


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract features from pose landmarks')

    parser.add_argument('--landmarks', type=str, required=True,
                        help='Path to landmark file (.npy) or directory containing landmark files')
    parser.add_argument('--output', type=str,
                        help='Output directory for extracted features')
    parser.add_argument('--config', type=str, default='default',
                        help='Base configuration file name (without .yaml extension)')
    parser.add_argument('--movement', type=str, default='squat',
                        help='Movement type for feature extraction (e.g., squat, ybt)')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata CSV file with video information')

    return parser.parse_args(args)


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

    if not movement_config:
        logging.error(f"Failed to load movement configuration: {args.movement}")
        sys.exit(1)

    # Merge configurations
    config = config_loader.merge_configs(args.config, args.movement)

    # Set default output directory if not specified
    if not args.output:
        args.output = base_config['paths']['data']['features']

    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Get feature extractor from pipeline
    feature_extractor = pipeline.get_component('feature_extractor')
    if not feature_extractor:
        logging.error(f"Failed to initialize feature extractor for movement type: {args.movement}")
        sys.exit(1)

    # Load metadata (command line takes priority over config)
    metadata_dict = None
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
    if metadata_path:
        from src.core.file_utils import load_video_metadata_file
        metadata_dict = load_video_metadata_file(metadata_path, config)
        if metadata_dict:
            logging.info(f"Loaded metadata for {len(metadata_dict)} videos")
            logging.info(f"Metadata includes videos: {', '.join(list(metadata_dict.keys())[:5])}...")
        else:
            logging.warning("Failed to load metadata or metadata file empty")
    else:
        logging.info("No metadata path available, proceeding without metadata")

    # Initialize DataFrame to collect all features
    all_features_df = pd.DataFrame()


    # Process single landmark file or directory of landmark files
    if os.path.isfile(args.landmarks) and args.landmarks.endswith('.npy'):
        # Process single landmarks file
        landmarks_path = args.landmarks
        video_id = Path(landmarks_path).stem.split('_')[0]  # Extract video ID from filename

        logging.info(f"Extracting features from landmarks: {landmarks_path}")

        # Load landmarks
        landmarks = load_landmarks(landmarks_path)
        if landmarks is None:
            logging.error(f"Failed to load landmarks from {landmarks_path}")
            sys.exit(1)

        # Get metadata for this video if available
        video_metadata = None
        if metadata_dict and video_id in metadata_dict:
            video_metadata = metadata_dict[video_id]
            video_metadata['video_id'] = video_id
            logging.info(f"Found metadata for video {video_id}")
        else:
            video_metadata = {'video_id': video_id}
            logging.info(f"No metadata found for video {video_id}, using default")

        # Extract features
        features_df = feature_extractor.extract_features(landmarks, video_metadata)

        if features_df is not None:
            # Save frame-by-frame data if available
            if 'frame_angles' in features_df.columns:
                frame_angles = features_df['frame_angles'].iloc[0]

                # Create directory for detailed outputs if needed
                detailed_output_dir = os.path.join(args.output, 'detailed', video_id)
                ensure_directory_exists(detailed_output_dir)

                # Save detailed frame angles
                frame_angles_df = pd.DataFrame({
                    'frame': range(len(frame_angles)),
                    'ankle_angle': frame_angles
                })
                frame_angles_file = os.path.join(detailed_output_dir, f"{video_id}_ankle_angles.csv")
                frame_angles_df.to_csv(frame_angles_file, index=False)
                logging.info(f"Saved detailed ankle angles to {frame_angles_file}")

                # Remove frame_angles column from features_df before saving aggregated features
                features_df = features_df.drop('frame_angles', axis=1)

            # Add to all features
            all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)

            # Save individual features file
            save_features(features_df, args.output, video_id)
            logging.info(f"Successfully extracted features from {landmarks_path}")
        else:
            logging.error(f"Failed to extract features from {landmarks_path}")

    elif os.path.isdir(args.landmarks):
        # Process all landmark files in directory
        landmark_files = list_files(args.landmarks, extension='.npy')
        logging.info(f"Found {len(landmark_files)} landmark files to process")

        success_count = 0
        for landmarks_path in landmark_files:
            video_id = Path(landmarks_path).stem.split('_')[0]  # Extract video ID from filename
            logging.info(f"Extracting features from landmarks: {landmarks_path}")

            # Load landmarks
            landmarks = load_landmarks(landmarks_path)
            if landmarks is None:
                logging.error(f"Failed to load landmarks from {landmarks_path}")
                continue

            # Get metadata for this video if available
            video_metadata = None
            if metadata_dict and video_id in metadata_dict:
                video_metadata = metadata_dict[video_id]
                video_metadata['video_id'] = video_id
                logging.info(f"Found metadata for video {video_id}")
            else:
                video_metadata = {'video_id': video_id}
                logging.info(f"No metadata found for video {video_id}, using default")

            # Extract features
            features_df = feature_extractor.extract_features(landmarks, video_metadata)

            if features_df is not None:
                # Save frame-by-frame data if available
                if 'frame_angles' in features_df.columns:
                    frame_angles = features_df['frame_angles'].iloc[0]

                    # Create directory for detailed outputs if needed
                    detailed_output_dir = os.path.join(args.output, 'detailed', video_id)
                    ensure_directory_exists(detailed_output_dir)

                    # Save detailed frame angles
                    frame_angles_df = pd.DataFrame({
                        'frame': range(len(frame_angles)),
                        'ankle_angle': frame_angles
                    })
                    frame_angles_file = os.path.join(detailed_output_dir, f"{video_id}_ankle_angles.csv")
                    frame_angles_df.to_csv(frame_angles_file, index=False)
                    logging.info(f"Saved detailed ankle angles to {frame_angles_file}")

                    # Remove frame_angles column from features_df before saving aggregated features
                    features_df = features_df.drop('frame_angles', axis=1)

                # Add to all features
                all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)

                # Save individual features file
                save_features(features_df, args.output, video_id)
                success_count += 1
                logging.info(f"Successfully extracted features from {landmarks_path}")
            else:
                logging.error(f"Failed to extract features from {landmarks_path}")

        logging.info(f"Extracted features from {success_count}/{len(landmark_files)} landmark files successfully")
    else:
        logging.error(f"Landmarks path is not a valid .npy file or directory: {args.landmarks}")
        sys.exit(1)

    # Save combined features if any were extracted
    if not all_features_df.empty:
        combined_features_file = os.path.join(args.output, f"{args.movement}_features.csv")
        all_features_df.to_csv(combined_features_file, index=False)
        logging.info(f"Saved combined features to {combined_features_file}")
    else:
        logging.warning("No features were extracted, combined features file not created")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    main()