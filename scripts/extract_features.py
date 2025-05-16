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

    parser.add_argument('--landmarks', type=str,
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


def process_landmarks_file(landmarks_path, output_path, feature_extractor, metadata_dict, all_features_df):
    """Process a single landmarks file"""
    video_id = Path(landmarks_path).stem.split('_')[0]  # Extract video ID from filename

    logging.info(f"Extracting features from landmarks: {landmarks_path}")

    # Load landmarks
    landmarks = load_landmarks(landmarks_path)
    if landmarks is None:
        logging.error(f"Failed to load landmarks from {landmarks_path}")
        return all_features_df, False

    # Get metadata for this video - try both with and without extension
    metadata_entry = None
    if metadata_dict and video_id in metadata_dict:
        metadata_entry = metadata_dict[video_id]
    elif metadata_dict and f"{video_id}.mp4" in metadata_dict:
        metadata_entry = metadata_dict[f"{video_id}.mp4"]

    if metadata_entry:
        video_metadata = metadata_entry.copy()
        video_metadata['video_id'] = video_id
        logging.info(f"Found metadata for video {video_id}")
    else:
        video_metadata = {'video_id': video_id}
        logging.info(f"No metadata found for video {video_id}, using default")

    # First, extract features for all frames for the detailed file
    full_features_df = feature_extractor.extract_features(landmarks, video_metadata)

    if full_features_df is None:
        logging.error(f"Failed to extract features from {landmarks_path}")
        return all_features_df, False

    # Save frame-by-frame data for all frames
    if 'frame_angles' in full_features_df.columns:
        frame_angles = full_features_df['frame_angles'].iloc[0]
        knee_frame_angles = full_features_df['knee_frame_angles'].iloc[0]
        hip_frame_angles = full_features_df['hip_flexion_frame_angles'].iloc[0]  # Get hip angles

        # Create directory for detailed outputs if needed
        detailed_output_dir = os.path.join(output_path, 'detailed', video_id)
        ensure_directory_exists(detailed_output_dir)

        # Save detailed frame angles for all frames
        frame_angles_df = pd.DataFrame({
            'frame': range(len(frame_angles)),
            'ankle_angle': frame_angles,
            'knee_angle': knee_frame_angles,
            'hip_flexion': hip_frame_angles  # Add hip flexion column
        })
        frame_angles_file = os.path.join(detailed_output_dir, f"{video_id}_ankle_angles.csv")
        frame_angles_df.to_csv(frame_angles_file, index=False)
        logging.info(f"Saved detailed ankle angles for all frames to {frame_angles_file}")

    # Now handle the aggregate statistics using repetition ranges
    if 'repetitions' in video_metadata:
        repetitions = video_metadata['repetitions']
        logging.info(f"Found {len(repetitions)} repetitions in metadata for {video_id}")

        # Create dataframe to hold aggregate stats for each repetition
        rep_stats = []

        for rep_idx, rep in enumerate(repetitions):
            start_frame = rep.get('start_frame', 0)
            end_frame = rep.get('end_frame', len(landmarks) - 1)

            # Adjust end_frame if it exceeds the landmark sequence length
            end_frame = min(end_frame, len(landmarks) - 1)

            logging.info(f"Analyzing repetition {rep_idx + 1}: frames {start_frame} to {end_frame}")

            # Extract angles for this repetition from the full set
            if 'frame_angles' in full_features_df.columns and len(full_features_df['frame_angles'].iloc[0]) > 0:
                rep_angles = full_features_df['frame_angles'].iloc[0][start_frame:end_frame + 1]
                rep_angles_knee = full_features_df['knee_frame_angles'].iloc[0][start_frame:end_frame + 1]
                rep_angles_hip = full_features_df['hip_flexion_frame_angles'].iloc[0][start_frame:end_frame + 1]  # Get hip angles for repetition

                if rep_angles:
                    # Calculate stats for this repetition
                    rep_stats.append({
                        'video_id': video_id,
                        'repetition': rep_idx + 1,
                        'ankle_angle_min': min(rep_angles),
                        'ankle_angle_max': max(rep_angles),
                        'ankle_angle_mean': sum(rep_angles) / len(rep_angles),
                        'knee_angle_min': min(rep_angles_knee),
                        'knee_angle_max': max(rep_angles_knee),
                        'knee_angle_mean': sum(rep_angles_knee) / len(rep_angles_knee),
                        'hip_flexion_min': min(rep_angles_hip),  # Add hip flexion stats
                        'hip_flexion_max': max(rep_angles_hip),
                        'hip_flexion_mean': sum(rep_angles_hip) / len(rep_angles_hip),
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })

        # Create aggregated features dataframe
        if rep_stats:
            features_df = pd.DataFrame(rep_stats)

            # Save individual features file with repetition stats
            save_features(features_df, output_path, video_id)
            logging.info(f"Successfully extracted features with {len(rep_stats)} repetitions from {landmarks_path}")

            # Add to all features
            updated_features_df = pd.concat([all_features_df, features_df], ignore_index=True)
            return updated_features_df, True
        else:
            logging.warning(f"No valid repetitions to analyze for {video_id}")
            return all_features_df, False
    else:
        # No repetitions in metadata, use stats from all frames
        logging.info(f"No repetitions found in metadata for {video_id}, using stats from all frames")

        # Remove frame_angles column from features_df before saving aggregated features
        if 'frame_angles' in full_features_df.columns:
            full_features_df = full_features_df.drop('frame_angles', axis=1)

        # Save individual features file
        save_features(full_features_df, output_path, video_id)
        logging.info(f"Successfully extracted features from all frames in {landmarks_path}")

        # Add to all features
        updated_features_df = pd.concat([all_features_df, full_features_df], ignore_index=True)
        return updated_features_df, True


def process_landmarks_directory(landmarks_dir, output_path, feature_extractor, metadata_dict, all_features_df):
    """Process all landmark files in a directory"""
    landmark_files = list_files(landmarks_dir, extension='.npy')
    logging.info(f"Found {len(landmark_files)} landmark files to process")

    success_count = 0
    for landmarks_path in landmark_files:
        all_features_df, success = process_landmarks_file(landmarks_path, output_path, feature_extractor, metadata_dict,
                                                          all_features_df)
        if success:
            success_count += 1

    logging.info(f"Extracted features from {success_count}/{len(landmark_files)} landmark files successfully")
    return all_features_df, success_count > 0


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
        if not args.landmarks:
            logging.error(
                "No metadata and no landmarks path provided. Either --landmarks or valid metadata is required.")
            sys.exit(1)

    # Initialize DataFrame to collect all features
    all_features_df = pd.DataFrame()

    # Process files based on landmarks path or metadata
    if args.landmarks:
        # Process landmarks from specified path
        if os.path.isfile(args.landmarks) and args.landmarks.endswith('.npy'):
            # Process single landmarks file
            all_features_df, _ = process_landmarks_file(args.landmarks, args.output, feature_extractor, metadata_dict,
                                                        all_features_df)
        elif os.path.isdir(args.landmarks):
            # Process all landmark files in directory
            all_features_df, _ = process_landmarks_directory(args.landmarks, args.output, feature_extractor,
                                                             metadata_dict, all_features_df)
        else:
            logging.error(f"Landmarks path is not a valid .npy file or directory: {args.landmarks}")
            sys.exit(1)
    elif metadata_dict:
        # Find and process landmarks for each video in metadata
        raw_video_dir = config.get('paths', {}).get('data', {}).get('raw', 'data/raw/')
        processed_video_dir = config.get('paths', {}).get('data', {}).get('processed_videos', {}).get('basic',
                                                                                                      'data/processed_videos/basic/')

        success_count = 0
        for video_id in metadata_dict.keys():
            # Construct the expected path to the landmarks file
            video_dir = os.path.join(processed_video_dir, Path(video_id).stem)
            landmarks_path = os.path.join(video_dir, f"{Path(video_id).stem}_mediapipe.npy")

            if os.path.exists(landmarks_path):
                logging.info(f"Found landmarks for {video_id} at {landmarks_path}")
                all_features_df, success = process_landmarks_file(landmarks_path, args.output, feature_extractor,
                                                                  metadata_dict, all_features_df)
                if success:
                    success_count += 1
            else:
                logging.warning(f"Landmarks file not found for {video_id} at expected path: {landmarks_path}")

        logging.info(f"Processed {success_count}/{len(metadata_dict)} videos successfully")
    else:
        logging.error("No landmarks path or valid metadata provided. Cannot proceed.")
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