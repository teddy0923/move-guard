# tests/test_file_utils.py
import os
import sys
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from core.file_utils import (
    ensure_directory_exists,
    save_landmarks,
    load_landmarks,
    save_features,
    load_features,
    save_model,
    load_model,
    save_metadata,
    load_metadata,
    list_files,
    load_video_metadata_file
)


class TestFileUtils:
    """Tests for the file utility functions"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_ensure_directory_exists(self, temp_dir):
        """Test creating directories"""
        # Create a nested directory
        nested_dir = os.path.join(temp_dir, 'a', 'b', 'c')
        result = ensure_directory_exists(nested_dir)

        # Verify directory was created
        assert result is True
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)

    def test_save_load_landmarks(self, temp_dir):
        """Test saving and loading landmarks"""
        # Create dummy landmarks data
        landmarks = np.random.random((10, 33, 3))  # 10 frames, 33 landmarks, 3 coordinates

        # Save landmarks
        filename = 'test_landmarks'
        saved_path = save_landmarks(landmarks, temp_dir, filename)

        # Verify file was saved
        assert saved_path is not None
        assert os.path.exists(saved_path)

        # Load landmarks
        loaded_landmarks = load_landmarks(saved_path)

        # Verify data matches
        assert loaded_landmarks is not None
        assert loaded_landmarks.shape == landmarks.shape
        assert np.array_equal(loaded_landmarks, landmarks)

    def test_save_load_features(self, temp_dir):
        """Test saving and loading features"""
        # Create dummy features data
        features = pd.DataFrame({
            'video_id': ['test1'],
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })

        # Save features
        filename = 'test_features'
        saved_path = save_features(features, temp_dir, filename)

        # Verify file was saved
        assert saved_path is not None
        assert os.path.exists(saved_path)

        # Load features
        loaded_features = load_features(saved_path)

        # Verify data matches
        assert loaded_features is not None
        assert loaded_features.shape == features.shape
        assert list(loaded_features.columns) == list(features.columns)
        assert loaded_features['video_id'].iloc[0] == features['video_id'].iloc[0]
        assert loaded_features['feature1'].iloc[0] == features['feature1'].iloc[0]

    def test_save_load_model(self, temp_dir):
        """Test saving and loading a model"""
        # Create a dummy model
        model = {
            'feature_names': ['f1', 'f2', 'f3'],
            'n_classes': 3,
            'classes': [1, 2, 3]
        }

        # Save model
        saved_path = save_model(model, temp_dir, 'test_model')

        # Verify file was saved
        assert saved_path is not None
        assert os.path.exists(saved_path)

        # Load model
        loaded_model = load_model(saved_path)

        # Verify data matches
        assert loaded_model is not None
        assert loaded_model['feature_names'] == model['feature_names']
        assert loaded_model['n_classes'] == model['n_classes']
        assert loaded_model['classes'] == model['classes']

    def test_save_load_metadata(self, temp_dir):
        """Test saving and loading metadata"""
        # Create dummy metadata
        metadata = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.80,
            'confusion_matrix': [[10, 2], [3, 9]]
        }

        # Save metadata
        saved_path = save_metadata(metadata, temp_dir, 'test_metadata')

        # Verify file was saved
        assert saved_path is not None
        assert os.path.exists(saved_path)

        # Load metadata
        loaded_metadata = load_metadata(saved_path)

        # Verify data matches
        assert loaded_metadata is not None
        assert loaded_metadata['accuracy'] == metadata['accuracy']
        assert loaded_metadata['confusion_matrix'] == metadata['confusion_matrix']

    def test_list_files(self, temp_dir):
        """Test listing files with extension filtering"""
        # Create test files
        for name in ['test1.txt', 'test2.txt', 'test3.csv', 'test4.csv', 'test5.npy']:
            with open(os.path.join(temp_dir, name), 'w') as f:
                f.write('test content')

        # List all files
        all_files = list_files(temp_dir)
        assert len(all_files) == 5

        # List with extension filter
        txt_files = list_files(temp_dir, extension='.txt')
        assert len(txt_files) == 2
        assert all(f.endswith('.txt') for f in txt_files)

        csv_files = list_files(temp_dir, extension='.csv')
        assert len(csv_files) == 2
        assert all(f.endswith('.csv') for f in csv_files)

        # List with multiple extension filter
        multi_ext = list_files(temp_dir, extension=('.txt', '.csv'))
        assert len(multi_ext) == 4
        assert all(f.endswith('.txt') or f.endswith('.csv') for f in multi_ext)

    def test_load_video_metadata_file(self, temp_dir):
        """Test loading video metadata from CSV"""
        # Create a test CSV file
        csv_path = os.path.join(temp_dir, 'video_metadata.csv')

        # Write test data
        with open(csv_path, 'w') as f:
            f.write("Filename,Movement,View_Angle,rep1_start,rep1_end,rep2_start,rep2_end\n")
            f.write("video1,squat,frontal,10,50,60,100\n")
            f.write("video2,squat,sagittal,15,45,55,85\n")

        # Load with default mapping
        metadata = load_video_metadata_file(csv_path)

        # Verify metadata is loaded
        assert metadata is not None
        assert 'video1' in metadata
        assert 'video2' in metadata

        # Verify content
        assert 'Movement' in metadata['video1']
        assert metadata['video1']['Movement'] == 'squat'
        assert 'View_Angle' in metadata['video1']
        assert metadata['video1']['View_Angle'] == 'frontal'

        # Verify rep data
        assert 'repetitions' in metadata['video1']
        assert len(metadata['video1']['repetitions']) == 2
        assert metadata['video1']['repetitions'][0]['start_frame'] == 10
        assert metadata['video1']['repetitions'][0]['end_frame'] == 50

        # Test with config-based mapping
        config = {
            'metadata': {
                'video_segments': {
                    'id_field': 'Filename',
                    'start_frame_fields': ['rep1_start', 'rep2_start'],
                    'end_frame_fields': ['rep1_end', 'rep2_end']
                }
            }
        }

        metadata_with_config = load_video_metadata_file(csv_path, config)

        # Verify config-based mapping works
        assert metadata_with_config is not None
        assert 'video1' in metadata_with_config
        assert 'repetitions' in metadata_with_config['video1']
        assert len(metadata_with_config['video1']['repetitions']) == 2

    def test_load_nonexistent_video_metadata(self):
        """Test loading non-existent video metadata file"""
        # Try to load non-existent file
        metadata = load_video_metadata_file('nonexistent_file.csv')

        # Verify result is None
        assert metadata is None