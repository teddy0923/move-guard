# scripts/extract_features.py
# !/usr/bin/env python3
"""
Script to extract features from pose landmarks for movement analysis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_landmarks, save_features, list_files
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

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Get feature extractor from pipeline
    feature_extractor = pipeline.get_component('feature_extractor')
    if not feature_extractor:
        logging.error(f"Failed to initialize feature extractor for movement type: {args.movement}")
        sys.exit(1)

    # Process single landmark file or directory of landmark files
    if os.path.isfile(args.landmarks) and args.landmarks.endswith('.npy'):
        # Process single landmarks file
        landmarks_path = args.landmarks
        output_path = args.output

        logging.info(f"Extracting features from landmarks: {landmarks_path}")

        # Load landmarks
        landmarks = load_landmarks(landmarks_path)
        if landmarks is None:
            logging.error(f"Failed to load landmarks from {landmarks_path}")
            sys.exit(1)

        # Extract features
        features = feature_extractor.extract_features(landmarks)

        if features is not None:
            # Save features
            filename = Path(landmarks_path).stem
            save_features(features, output_path, filename)
            logging.info(f"Successfully extracted features from {landmarks_path}")
        else:
            logging.error(f"Failed to extract features from {landmarks_path}")

    elif os.path.isdir(args.landmarks):
        # Process all landmark files in directory
        landmark_files = list_files(args.landmarks, extension='.npy')
        logging.info(f"Found {len(landmark_files)} landmark files to process")

        success_count = 0
        for landmarks_path in landmark_files:
            logging.info(f"Extracting features from landmarks: {landmarks_path}")

            # Load landmarks
            landmarks = load_landmarks(landmarks_path)
            if landmarks is None:
                logging.error(f"Failed to load landmarks from {landmarks_path}")
                continue

            # Extract features
            features = feature_extractor.extract_features(landmarks)

            if features is not None:
                # Save features
                filename = Path(landmarks_path).stem
                save_features(features, args.output, filename)
                success_count += 1
                logging.info(f"Successfully extracted features from {landmarks_path}")
            else:
                logging.error(f"Failed to extract features from {landmarks_path}")

        logging.info(f"Extracted features from {success_count}/{len(landmark_files)} landmark files successfully")
    else:
        logging.error(f"Landmarks path is not a valid .npy file or directory: {args.landmarks}")
        sys.exit(1)


if __name__ == "__main__":
    main()