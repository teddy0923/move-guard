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
import numpy as np

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
    if 'ankle_frame_angles' in full_features_df.columns:
        ankle_angles = full_features_df['ankle_frame_angles'].iloc[0] or []
        knee_angles = full_features_df['knee_frame_angles'].iloc[0] or []
        hip_angles = full_features_df['hip_flexion_frame_angles'].iloc[0] or []
        shoulder_angles = full_features_df['shoulder_flexion_frame_angles'].iloc[0] or []
        femoral_angles = full_features_df['femoral_frame_angles'].iloc[0] or []
        trunk_tibia_angles = full_features_df['trunk_tibia_angle_frame_angles'].iloc[0] or []

        # Create a DataFrame with all angles for each frame
        frame_angles_dict = {
            'frame': range(max(len(arr) for arr in [
                ankle_angles, knee_angles, hip_angles,
                shoulder_angles, femoral_angles, trunk_tibia_angles
            ])),
            'ankle_angle': ankle_angles,
            'knee_angle': knee_angles,
            'hip_flexion': hip_angles,
            'shoulder_flexion': shoulder_angles,
            'femoral_angle': femoral_angles,
            'trunk_tibia_angle': trunk_tibia_angles
        }

        # Create directory for detailed outputs if needed
        detailed_output_dir = os.path.join(output_path, 'detailed', video_id)
        ensure_directory_exists(detailed_output_dir)

        # Pad shorter arrays with None
        max_len = len(frame_angles_dict['frame'])
        for key in frame_angles_dict:
            if key != 'frame':
                frame_angles_dict[key] = list(frame_angles_dict[key]) + [None] * (max_len - len(frame_angles_dict[key]))

        # Save detailed frame angles for all frames
        frame_angles_df = pd.DataFrame(frame_angles_dict)
        frame_angles_file = os.path.join(detailed_output_dir, f"{video_id}_angles.csv")
        frame_angles_df.to_csv(frame_angles_file, index=False)
        logging.info(f"Saved detailed angles for all frames to {frame_angles_file}")

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
            if 'ankle_frame_angles' in full_features_df.columns:
                # Get the angle arrays
                ankle_angles = full_features_df['ankle_frame_angles'].iloc[0]
                knee_angles = full_features_df['knee_frame_angles'].iloc[0]
                hip_angles = full_features_df['hip_flexion_frame_angles'].iloc[0]
                shoulder_angles = full_features_df['shoulder_flexion_frame_angles'].iloc[0]
                femoral_angles = full_features_df['femoral_frame_angles'].iloc[0]
                trunk_tibia_angles = full_features_df['trunk_tibia_angle_frame_angles'].iloc[0]

                # Extract angles for this repetition
                rep_angles = ankle_angles[start_frame:end_frame + 1]
                rep_knee_angles = knee_angles[start_frame:end_frame + 1]
                rep_hip_angles = hip_angles[start_frame:end_frame + 1]
                rep_shoulder_angles = shoulder_angles[start_frame:end_frame + 1]
                rep_femoral_angles = femoral_angles[start_frame:end_frame + 1]
                rep_trunk_tibia_angles = trunk_tibia_angles[start_frame:end_frame + 1]

                if len(rep_angles) > 0:
                    # Calculate min and max values
                    ankle_min = min(rep_angles)
                    ankle_max = max(rep_angles)
                    knee_min = min(rep_knee_angles)
                    knee_max = max(rep_knee_angles)
                    hip_min = min(rep_hip_angles)
                    hip_max = max(rep_hip_angles)
                    shoulder_min = min(rep_shoulder_angles)
                    shoulder_max = max(rep_shoulder_angles)
                    femoral_min = min(rep_femoral_angles)
                    femoral_max = max(rep_femoral_angles)
                    trunk_tibia_min = min(rep_trunk_tibia_angles)
                    trunk_tibia_max = max(rep_trunk_tibia_angles)

                    # Calculate stats for this repetition
                    rep_features = {
                        'video_id': video_id,
                        'repetition': rep_idx + 1,
                        'ankle_angle_min': ankle_min,
                        'ankle_angle_max': ankle_max,
                        'ankle_angle_mean': sum(rep_angles) / len(rep_angles),
                        'ankle_angle_range': ankle_max - ankle_min,
                        'knee_angle_min': knee_min,
                        'knee_angle_max': knee_max,
                        'knee_angle_mean': sum(rep_knee_angles) / len(rep_knee_angles),
                        'knee_angle_range': knee_max - knee_min,
                        'hip_flexion_min': hip_min,
                        'hip_flexion_max': hip_max,
                        'hip_flexion_mean': sum(rep_hip_angles) / len(rep_hip_angles),
                        'hip_flexion_range': hip_max - hip_min,
                        'shoulder_flexion_min': shoulder_min,
                        'shoulder_flexion_max': shoulder_max,
                        'shoulder_flexion_mean': sum(rep_shoulder_angles) / len(rep_shoulder_angles),
                        'shoulder_flexion_range': shoulder_max - shoulder_min,
                        'femoral_angle_min': femoral_min,
                        'femoral_angle_max': femoral_max,
                        'femoral_angle_mean': sum(rep_femoral_angles) / len(rep_femoral_angles),
                        'femoral_angle_range': femoral_max - femoral_min,
                        'trunk_tibia_angle_min': trunk_tibia_min,
                        'trunk_tibia_angle_max': trunk_tibia_max,
                        'trunk_tibia_angle_mean': sum(rep_trunk_tibia_angles) / len(rep_trunk_tibia_angles),
                        'trunk_tibia_angle_range': trunk_tibia_max - trunk_tibia_min,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    }

                    rep_stats.append(rep_features)

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


def process_front_landmarks_file(landmarks_path, feature_extractor, metadata_dict, all_front_features_df):
    """Process a single landmarks file for front view"""
    try:
        # Load landmarks
        landmarks_sequence = np.load(landmarks_path)
        video_id = Path(landmarks_path).stem.replace('_mediapipe', '')

        # Try different ways to find metadata
        video_metadata = None
        possible_ids = [
            video_id,                # Try without extension
            video_id + '.mp4',      # Try with .mp4
            Path(video_id).stem     # Try just the stem
        ]

        for possible_id in possible_ids:
            if possible_id in metadata_dict:
                video_metadata = metadata_dict[possible_id]
                video_metadata['video_id'] = video_id  # Ensure video_id is in metadata
                logging.info(f"Found metadata using ID: {possible_id}")
                break

        # Check if this is a front view video
        if video_metadata:
            # Try both View_Angle and view_angle
            view_angle = video_metadata.get('View_Angle', video_metadata.get('view_angle', '')).lower()
            logging.info(f"Found view angle: {view_angle} for {video_id}")
            if 'front' in view_angle:
                logging.info(f"Processing front view for {video_id}")
            else:
                logging.info(f"Skipping front view processing for {video_id} as it is not a front view video (view_angle: {view_angle})")
                return all_front_features_df, False
        else:
            logging.info(f"No metadata found for {video_id}, tried IDs: {possible_ids}")
            logging.info(f"Available metadata keys: {list(metadata_dict.keys())}")
            return all_front_features_df, False

        # Extract features
        front_features_df = feature_extractor.extract_front_features(landmarks_sequence, video_metadata)

        if front_features_df is not None and not front_features_df.empty:
            # Create the detailed output directory if it doesn't exist
            detailed_output_dir = os.path.join('data', 'features', 'detailed', video_id)
            os.makedirs(detailed_output_dir, exist_ok=True)

            # Save the frame-by-frame data
            detailed_output_path = os.path.join(detailed_output_dir, f"{video_id}_front_angles.csv")
            front_features_df.to_csv(detailed_output_path, index=False)
            logging.info(f"Saved detailed front view features to {detailed_output_path}")

            # Calculate per-repetition statistics
            repetitions = video_metadata.get('repetitions', [])
            if repetitions:
                logging.info(f"Found {len(repetitions)} repetitions in metadata for {video_id}")
                rep_stats = []
                for rep in repetitions:
                    start_frame = rep['start_frame']
                    end_frame = rep['end_frame']
                    logging.info(f"Analyzing repetition {rep['rep_num']}: frames {start_frame} to {end_frame}")
                    
                    rep_data = front_features_df[(front_features_df['frame'] >= start_frame) & 
                                              (front_features_df['frame'] <= end_frame)]
                    
                    # Calculate statistics for left elbow
                    left_min = rep_data['left_elbow_angle'].min()
                    left_max = rep_data['left_elbow_angle'].max()
                    left_mean = rep_data['left_elbow_angle'].mean()
                    left_range = left_max - left_min
                    
                    # Calculate statistics for right elbow
                    right_min = rep_data['right_elbow_angle'].min()
                    right_max = rep_data['right_elbow_angle'].max()
                    right_mean = rep_data['right_elbow_angle'].mean()
                    right_range = right_max - right_min
                    
                    # Calculate statistics for hip alignment
                    hip_min = rep_data['hip_alignment'].min()
                    hip_max = rep_data['hip_alignment'].max()
                    hip_mean = rep_data['hip_alignment'].mean()
                    hip_range = hip_max - hip_min
                    
                    # Calculate statistics for shoulder alignment
                    shoulder_min = rep_data['shoulder_alignment'].min()
                    shoulder_max = rep_data['shoulder_alignment'].max()
                    shoulder_mean = rep_data['shoulder_alignment'].mean()
                    shoulder_range = shoulder_max - shoulder_min
                    
                    # Calculate statistics for knee-ankle ratio
                    knee_ankle_min = rep_data['knee_ankle_ratio'].min()
                    knee_ankle_max = rep_data['knee_ankle_ratio'].max()
                    knee_ankle_mean = rep_data['knee_ankle_ratio'].mean()
                    knee_ankle_range = knee_ankle_max - knee_ankle_min
                    
                    # Calculate statistics for knee distance
                    knee_dist_min = rep_data['knee_distance'].min()
                    knee_dist_max = rep_data['knee_distance'].max()
                    knee_dist_mean = rep_data['knee_distance'].mean()
                    knee_dist_range = knee_dist_max - knee_dist_min
                    
                    # Calculate statistics for ankle distance
                    ankle_dist_min = rep_data['ankle_distance'].min()
                    ankle_dist_max = rep_data['ankle_distance'].max()
                    ankle_dist_mean = rep_data['ankle_distance'].mean()
                    ankle_dist_range = ankle_dist_max - ankle_dist_min
                    
                    rep_stats.append({
                        'video_id': video_id,
                        'rep_num': rep['rep_num'],
                        'left_elbow_angle_min': left_min,
                        'left_elbow_angle_max': left_max,
                        'left_elbow_angle_mean': left_mean,
                        'left_elbow_angle_range': left_range,
                        'right_elbow_angle_min': right_min,
                        'right_elbow_angle_max': right_max,
                        'right_elbow_angle_mean': right_mean,
                        'right_elbow_angle_range': right_range,
                        'hip_alignment_min': hip_min,
                        'hip_alignment_max': hip_max,
                        'hip_alignment_mean': hip_mean,
                        'hip_alignment_range': hip_range,
                        'shoulder_alignment_min': shoulder_min,
                        'shoulder_alignment_max': shoulder_max,
                        'shoulder_alignment_mean': shoulder_mean,
                        'shoulder_alignment_range': shoulder_range,
                        'knee_distance_min': knee_dist_min,
                        'knee_distance_max': knee_dist_max,
                        'knee_distance_mean': knee_dist_mean,
                        'knee_distance_range': knee_dist_range,
                        'ankle_distance_min': ankle_dist_min,
                        'ankle_distance_max': ankle_dist_max,
                        'ankle_distance_mean': ankle_dist_mean,
                        'ankle_distance_range': ankle_dist_range,
                        'knee_ankle_ratio_min': knee_ankle_min,
                        'knee_ankle_ratio_max': knee_ankle_max,
                        'knee_ankle_ratio_mean': knee_ankle_mean,
                        'knee_ankle_ratio_range': knee_ankle_range
                    })
                
                # Create and save summary DataFrame for this video
                summary_df = pd.DataFrame(rep_stats)
                summary_path = os.path.join('data', 'features', f"{video_id}.csv")
                summary_df.to_csv(summary_path, index=False)
                logging.info(f"Saved summary statistics to {summary_path}")
                
                # Update or create the aggregate features file
                aggregate_path = os.path.join('data', 'features', 'squat_features_front.csv')
                if os.path.exists(aggregate_path):
                    # Read existing aggregate file
                    aggregate_df = pd.read_csv(aggregate_path)
                    # Remove any existing entries for this video
                    aggregate_df = aggregate_df[aggregate_df['video_id'] != video_id]
                    # Append new data
                    aggregate_df = pd.concat([aggregate_df, summary_df], ignore_index=True)
                else:
                    # Create new aggregate file
                    aggregate_df = summary_df.copy()
                
                # Save updated aggregate file
                aggregate_df.to_csv(aggregate_path, index=False)
                logging.info(f"Updated aggregate features file at {aggregate_path}")

            # Update the all_features DataFrame
            updated_front_features_df = pd.concat([all_front_features_df, front_features_df], ignore_index=True)
            return updated_front_features_df, True
        else:
            logging.warning(f"No front features extracted for {video_id}")
            return all_front_features_df, False

    except Exception as e:
        logging.error(f"Error processing front landmarks file {landmarks_path}: {str(e)}")
        return all_front_features_df, False


def process_front_landmarks_directory(landmarks_dir, feature_extractor, metadata_dict, all_front_features_df):
    """Process all landmark files in a directory for front view"""
    landmark_files = list_files(landmarks_dir, extension='.npy')
    logging.info(f"Found {len(landmark_files)} landmark files to process for front view")

    success_count = 0
    for landmarks_path in landmark_files:
        all_front_features_df, success = process_front_landmarks_file(landmarks_path, feature_extractor, metadata_dict,
                                                                    all_front_features_df)
        if success:
            success_count += 1

    logging.info(f"Extracted front view features from {success_count}/{len(landmark_files)} landmark files successfully")
    return all_front_features_df, success_count > 0


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
    all_front_features_df = pd.DataFrame()  # Initialize front features DataFrame

    # Process files based on landmarks path or metadata
    if args.landmarks:
        # Process landmarks from specified path
        if os.path.isfile(args.landmarks) and args.landmarks.endswith('.npy'):
            # Process single landmarks file
            all_features_df, _ = process_landmarks_file(args.landmarks, args.output, feature_extractor, metadata_dict,
                                                        all_features_df)
            # Process front view features but don't save
            all_front_features_df, _ = process_front_landmarks_file(args.landmarks, feature_extractor, metadata_dict,
                                                                  all_front_features_df)
        elif os.path.isdir(args.landmarks):
            # Process all landmark files in directory
            all_features_df, _ = process_landmarks_directory(args.landmarks, args.output, feature_extractor,
                                                             metadata_dict, all_features_df)
            # Process all front landmark files in directory
            all_front_features_df, _ = process_front_landmarks_directory(args.landmarks, feature_extractor, metadata_dict,
                                                                       all_front_features_df)
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
                # Process front view features but don't save
                all_front_features_df, front_success = process_front_landmarks_file(landmarks_path, feature_extractor,
                                                                                metadata_dict, all_front_features_df)
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