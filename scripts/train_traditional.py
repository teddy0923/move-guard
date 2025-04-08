# scripts/train_traditional.py
# !/usr/bin/env python3
"""
Script to train machine learning model on extracted features for movement quality classification.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigLoader
from core.file_utils import list_files, save_model, save_metadata
from core.pipeline import Pipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ML model on movement features')

    parser.add_argument('--features', type=str, required=True,
                        help='Path to features file (.csv) or directory containing feature files')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels file (.csv) with movement quality annotations')
    parser.add_argument('--output', type=str,
                        help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')
    parser.add_argument('--movement', type=str, default='squat',
                        help='Movement type (e.g., squat, ybt)')

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

    # Load movement-specific configuration if needed
    if args.movement != 'default':
        movement_config = config_loader.load_config(args.movement)
        if movement_config:
            config = config_loader.merge_configs(args.config, args.movement)

    # Set default output directory if not specified
    if not args.output:
        args.output = os.path.join(config.get('evaluation', {}).get('results_path', 'results/'), args.movement)

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Load features
    if os.path.isfile(args.features) and args.features.endswith('.csv'):
        # Load single features file
        features_df = pd.read_csv(args.features)
    elif os.path.isdir(args.features):
        # Combine multiple feature files
        feature_files = list_files(args.features, extension='.csv')
        if not feature_files:
            logging.error(f"No feature files found in directory: {args.features}")
            sys.exit(1)

        feature_dfs = []
        for file_path in feature_files:
            try:
                df = pd.read_csv(file_path)
                feature_dfs.append(df)
            except Exception as e:
                logging.error(f"Error loading features from {file_path}: {str(e)}")

        if not feature_dfs:
            logging.error("Failed to load any feature files")
            sys.exit(1)

        features_df = pd.concat(feature_dfs, ignore_index=True)
    else:
        logging.error(f"Features path is not a valid .csv file or directory: {args.features}")
        sys.exit(1)

    # Load labels
    try:
        labels_df = pd.read_csv(args.labels)
    except Exception as e:
        logging.error(f"Error loading labels from {args.labels}: {str(e)}")
        sys.exit(1)

    # Merge features with labels
    try:
        # First, try to find common columns
        common_cols = set(features_df.columns).intersection(set(labels_df.columns))
        common_cols = [col for col in common_cols if col not in ['label', 'quality', 'class']]

        if not common_cols:
            logging.error("No common columns found between features and labels")
            sys.exit(1)

        # Use the first common column as the join key
        join_col = common_cols[0]
        logging.info(f"Joining features and labels on column: {join_col}")

        # Find the label column (assumed to be 'label', 'quality', or 'class')
        label_col = next((col for col in ['label', 'quality', 'class'] if col in labels_df.columns), None)
        if not label_col:
            logging.error("No label column found in labels file")
            sys.exit(1)

        # Join features and labels
        combined_df = pd.merge(features_df, labels_df[[join_col, label_col]], on=join_col)

        features = combined_df.drop(columns=[join_col, label_col])
        labels = combined_df[label_col]
    except Exception as e:
        logging.error(f"Error merging features and labels: {str(e)}")
        sys.exit(1)

    # Train model using pipeline
    logging.info("Training model...")
    training_result = pipeline.train_model(features, labels)

    if training_result['success']:
        # Get model from pipeline
        ml_model = pipeline.get_component('ml_model')

        # Save model
        model_path = save_model(ml_model.model, args.output, f"{args.movement}_model")

        # Get feature importance
        importance = ml_model.feature_importance()
        if importance:
            importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

            importance_df.to_csv(os.path.join(args.output, f"{args.movement}_feature_importance.csv"), index=False)

        # Save training metadata
        metadata = {
            'movement_type': args.movement,
            'features_shape': features.shape,
            'class_distribution': labels.value_counts().to_dict(),
            'training_metrics': training_result.get('training_metrics', {}),
            'feature_importance': importance if importance else None
        }

        save_metadata(metadata, args.output, f"{args.movement}_training_metadata")

        logging.info(f"Model training completed successfully. Model saved to {model_path}")
    else:
        logging.error(f"Model training failed: {training_result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()