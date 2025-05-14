# core/file_utils.py
import os
import shutil
import json
import pickle
import logging
import numpy as np
import csv
from pathlib import Path


def ensure_directory_exists(directory_path):
    """
    Create directory if it doesn't exist

    Args:
        directory_path: Path to the directory

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {str(e)}")
        return False


def save_landmarks(landmarks, output_path, filename, estimator_name, metadata=None):
    """
    Save extracted landmarks to a numpy file, with optional metadata

    Args:
        landmarks: Numpy array of landmarks
        output_path: Directory to save the file
        filename: Name of the file without extension
        estimator_name: Name of the pose estimator algorithm
        metadata: Optional dictionary with metadata to save alongside landmarks

    Returns:
        str: Path to the saved file or None if error
    """
    ensure_directory_exists(output_path)
    try:
        file_path = os.path.join(output_path, f"{filename}_{estimator_name}.npy")
        np.save(file_path, landmarks)
        logging.info(f"Landmarks saved to {file_path}")

        # Save metadata if provided
        if metadata:
            metadata_file = os.path.join(output_path, f"{filename}_{estimator_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info(f"Landmark metadata saved to {metadata_file}")

        return file_path
    except Exception as e:
        logging.error(f"Error saving landmarks: {str(e)}")
        return None

def load_landmarks(file_path):
    """
    Load landmarks from a numpy file

    Args:
        file_path: Path to the landmark file

    Returns:
        Numpy array of landmarks or None if error
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Landmark file not found: {file_path}")
            return None

        landmarks = np.load(file_path, allow_pickle=True)
        return landmarks
    except Exception as e:
        logging.error(f"Error loading landmarks from {file_path}: {str(e)}")
        return None


def save_features(features, output_path, filename):
    """
    Save extracted features to a CSV file

    Args:
        features: DataFrame of features
        output_path: Directory to save the file
        filename: Name of the file without extension

    Returns:
        str: Path to the saved file or None if error
    """
    ensure_directory_exists(output_path)
    try:
        file_path = os.path.join(output_path, f"{filename}.csv")
        features.to_csv(file_path, index=False)
        logging.info(f"Features saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving features: {str(e)}")
        return None


def load_features(file_path):
    """
    Load features from a CSV file

    Args:
        file_path: Path to the feature file

    Returns:
        DataFrame of features or None if error
    """
    try:
        import pandas as pd
        if not os.path.exists(file_path):
            logging.error(f"Feature file not found: {file_path}")
            return None

        features = pd.read_csv(file_path)
        return features
    except Exception as e:
        logging.error(f"Error loading features from {file_path}: {str(e)}")
        return None


def save_model(model, output_path, filename):
    """
    Save trained model to a pickle file

    Args:
        model: Trained ML model
        output_path: Directory to save the file
        filename: Name of the file without extension

    Returns:
        str: Path to the saved file or None if error
    """
    ensure_directory_exists(output_path)
    try:
        file_path = os.path.join(output_path, f"{filename}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return None


def load_model(file_path):
    """
    Load model from a pickle file

    Args:
        file_path: Path to the model file

    Returns:
        Trained ML model or None if error
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Model file not found: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {str(e)}")
        return None


def save_metadata(metadata, output_path, filename):
    """
    Save metadata to a JSON file

    Args:
        metadata: Dictionary of metadata
        output_path: Directory to save the file
        filename: Name of the file without extension

    Returns:
        str: Path to the saved file or None if error
    """
    ensure_directory_exists(output_path)
    try:
        file_path = os.path.join(output_path, f"{filename}.json")
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving metadata: {str(e)}")
        return None


def load_metadata(file_path):
    """
    Load metadata from a JSON file

    Args:
        file_path: Path to the metadata file

    Returns:
        Dictionary of metadata or None if error
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Metadata file not found: {file_path}")
            return None

        with open(file_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata from {file_path}: {str(e)}")
        return None


def list_files(directory, extension=None):
    """
    List all files in a directory with optional extension filter

    Args:
        directory: Directory to list files from
        extension: Optional file extension to filter by (e.g., '.mp4')

    Returns:
        list: List of file paths
    """
    try:
        if not os.path.exists(directory):
            logging.warning(f"Directory not found: {directory}")
            return []

        if extension:
            return [os.path.join(directory, f) for f in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
        else:
            return [os.path.join(directory, f) for f in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        logging.error(f"Error listing files in {directory}: {str(e)}")
        return []


def get_video_metadata(video_path):
    """
    Extract metadata from a video file

    Args:
        video_path: Path to the video file

    Returns:
        dict: Dictionary with video metadata or None if error
    """
    try:
        import cv2
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return None

        # Extract video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'file_path': video_path,
            'file_name': os.path.basename(video_path),
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'file_size_bytes': os.path.getsize(video_path)
        }
    except Exception as e:
        logging.error(f"Error extracting video metadata from {video_path}: {str(e)}")
        return None


def path_from_config(config, *keys):
    """
    Build file path from configuration

    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to navigate the config dictionary

    Returns:
        Path: Constructed path object or None if keys are invalid
    """
    try:
        value = config
        for key in keys:
            value = value[key]

        return Path(value)
    except (KeyError, TypeError) as e:
        logging.error(f"Error constructing path from config keys {keys}: {str(e)}")
        return None


def load_video_metadata_file(metadata_path, config=None):
    """
    Load video metadata from a CSV file containing start/end frames

    Args:
        metadata_path: Path to the metadata CSV file
        config: Optional configuration dictionary with metadata field mappings

    Returns:
        dict: Dictionary mapping video IDs to their metadata (start/end frames)
              or None if metadata file doesn't exist or can't be loaded
    """
    if not metadata_path or not os.path.exists(metadata_path):
        return None

    try:
        import pandas as pd
        metadata_df = pd.read_csv(metadata_path)

        # Default field mappings
        default_mappings = {
            'id_field': 'video_id',
            'start_frame_fields': ['start_frame'],
            'end_frame_fields': ['end_frame']
        }

        # Get field mappings from config if provided
        field_mappings = default_mappings
        if config and 'metadata' in config and 'video_segments' in config['metadata']:
            field_mappings = config['metadata']['video_segments']

        # Extract the ID field name (filename or video_id)
        id_field = field_mappings.get('id_field', 'video_id')

        # Handle case where the ID field doesn't exist
        if id_field not in metadata_df.columns:
            logging.warning(
                f"ID field '{id_field}' not found in metadata. Available columns: {metadata_df.columns.tolist()}")
            # Try to use the first column as ID if it exists
            if len(metadata_df.columns) > 0:
                id_field = metadata_df.columns[0]
                logging.warning(f"Using '{id_field}' as ID field instead")
            else:
                return None

        # Create metadata dictionary
        metadata_dict = {}
        for _, row in metadata_df.iterrows():
            video_id = row.get(id_field) or os.path.basename(row.get('video_path', ''))

            # Skip entries without valid video identification
            if not video_id:
                continue

            # Initialize metadata for this video
            video_metadata = {}

            # First add any other metadata columns (non-repetition data)
            for col in metadata_df.columns:
                if col != id_field and pd.notna(row[col]):
                    video_metadata[col] = row[col]

            # Find repetition columns dynamically
            rep_columns = metadata_df.columns.tolist()
            repetitions = []

            # Look for columns that follow the pattern rep#_start and rep#_end
            import re
            start_pattern = re.compile(r'rep(\d+)_start')

            for col in rep_columns:
                start_match = start_pattern.match(col)
                if start_match and pd.notna(row[col]):
                    rep_num = start_match.group(1)
                    end_col = f'rep{rep_num}_end'

                    # Only add if both start and end values exist and are valid
                    if end_col in rep_columns and pd.notna(row[end_col]):
                        repetitions.append({
                            'rep_num': int(rep_num),
                            'start_frame': int(row[col]),
                            'end_frame': int(row[end_col])
                        })

            # Sort repetitions by rep_num
            repetitions.sort(key=lambda x: x['rep_num'])

            # If there are repetitions, add them to metadata
            if repetitions:
                video_metadata['repetitions'] = repetitions
                # Use first repetition as default start/end frames
                video_metadata['start_frame'] = repetitions[0]['start_frame']
                video_metadata['end_frame'] = repetitions[0]['end_frame']

            # Add this video's metadata to the dictionary
            metadata_dict[video_id] = video_metadata

        return metadata_dict
    except Exception as e:
        logging.error(f"Error loading video metadata from {metadata_path}: {str(e)}")
        return None

def save_video_metadata(metadata, output_path, filename):
    """Save video metadata alongside landmark data"""
    metadata_file = os.path.join(output_path, f"{filename}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata_file
