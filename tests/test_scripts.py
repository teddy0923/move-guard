# tests/test_scripts.py
import os
import sys
import pytest
from pathlib import Path
import tempfile
import shutil
import importlib.util

# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))


# Helper function to load script module
def load_script(script_name):
    """Load a script file as a module"""
    script_path = os.path.join(project_root, 'scripts', f"{script_name}.py")
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScripts:
    """Tests for script functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_video(self, temp_dir):
        """Create a test video file"""
        # Create a dummy video file
        video_path = os.path.join(temp_dir, 'test_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(b'dummy video content')

        return video_path

    @pytest.fixture
    def test_landmarks(self, temp_dir):
        """Create test landmarks file"""
        import numpy as np

        # Create landmarks directory
        landmarks_dir = os.path.join(temp_dir, 'landmarks')
        os.makedirs(landmarks_dir, exist_ok=True)

        # Create dummy landmarks data
        landmarks = np.random.random((10, 33, 3))  # 10 frames, 33 landmarks, 3 coordinates

        # Save landmarks
        landmarks_path = os.path.join(landmarks_dir, 'test_video.npy')
        np.save(landmarks_path, landmarks)

        return landmarks_path

    @pytest.fixture
    def test_features(self, temp_dir):
        """Create test features file"""
        import pandas as pd

        # Create features directory
        features_dir = os.path.join(temp_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        # Create dummy features data
        features = pd.DataFrame({
            'video_id': ['test_video'],
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })

        # Save features
        features_path = os.path.join(features_dir, 'test_video.csv')
        features.to_csv(features_path, index=False)

        return features_path

    @pytest.fixture
    def test_labels(self, temp_dir):
        """Create test labels file"""
        import pandas as pd

        # Create labels directory
        labels_dir = os.path.join(temp_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        # Create dummy labels data
        labels = pd.DataFrame({
            'video_id': ['test_video'],
            'quality': [2]  # Moderate quality
        })

        # Save labels
        labels_path = os.path.join(labels_dir, 'test_labels.csv')
        labels.to_csv(labels_path, index=False)

        return labels_path

    @pytest.fixture
    def test_model(self, temp_dir):
        """Create test model file"""
        import pickle

        # Create models directory
        models_dir = os.path.join(temp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Create dummy model
        model = {
            'feature_names': ['feature1', 'feature2', 'feature3'],
            'n_classes': 3,
            'classes': [1, 2, 3]
        }

        # Save model
        model_path = os.path.join(models_dir, 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return model_path

    @pytest.fixture
    def test_metadata(self, temp_dir):
        """Create test metadata file"""
        import pandas as pd

        # Create metadata directory
        metadata_dir = os.path.join(temp_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        # Create dummy metadata CSV
        metadata = pd.DataFrame({
            'Filename': ['test_video'],
            'Movement': ['squat'],
            'View_Angle': ['frontal'],
            'rep1_start': [10],
            'rep1_end': [50],
            'rep2_start': [60],
            'rep2_end': [100]
        })

        # Save metadata
        metadata_path = os.path.join(metadata_dir, 'test_metadata.csv')
        metadata.to_csv(metadata_path, index=False)

        return metadata_path

    def test_process_video_arg_parsing(self):
        """Test process_video.py argument parsing"""
        # Load the script as a module
        process_video = load_script('process_video')

        # Test argument parsing
        args = process_video.parse_args(['--video', 'test.mp4'])

        # Verify required args
        assert args.video == 'test.mp4'
        assert args.config == 'default'  # Default value

    def test_extract_features_arg_parsing(self):
        """Test extract_features.py argument parsing"""
        # Load the script as a module
        extract_features = load_script('extract_features')

        # Test argument parsing
        args = extract_features.parse_args(['--landmarks', 'landmarks.npy'])

        # Verify required args
        assert args.landmarks == 'landmarks.npy'
        assert args.config == 'default'  # Default value
        assert args.movement == 'squat'  # Default value

    def test_train_model_arg_parsing(self):
        """Test train_traditional.py argument parsing"""
        # Load the script as a module
        train_model = load_script('train_traditional')

        # Test argument parsing
        args = train_model.parse_args(['--features', 'features.csv', '--labels', 'labels.csv'])

        # Verify required args
        assert args.features == 'features.csv'
        assert args.labels == 'labels.csv'
        assert args.config == 'default'  # Default value
        assert args.movement == 'squat'  # Default value

    def test_evaluate_model_arg_parsing(self):
        """Test evaluate_model.py argument parsing"""
        # Load the script as a module
        evaluate_model = load_script('evaluate_model')

        # Test argument parsing
        args = evaluate_model.parse_args([
            '--features', 'features.csv',
            '--labels', 'labels.csv',
            '--model', 'model.pkl'
        ])

        # Verify required args
        assert args.features == 'features.csv'
        assert args.labels == 'labels.csv'
        assert args.model == 'model.pkl'
        assert args.config == 'default'  # Default value

    def test_generate_visualizations_arg_parsing(self):
        """Test generate_visualizations.py argument parsing"""
        # Load the script as a module
        generate_visualizations = load_script('generate_visualizations')

        # Test argument parsing
        args = generate_visualizations.parse_args([
            '--video', 'video.mp4',
            '--landmarks', 'landmarks.npy'
        ])

        # Verify required args
        assert args.video == 'video.mp4'
        assert args.landmarks == 'landmarks.npy'
        assert args.config == 'default'  # Default value
        assert args.movement == 'squat'  # Default value
        assert args.angles is False  # Default value

    def test_mediapipe_estimator():
        """Test the MediaPipe pose estimator implementation"""
        from tests.test_mediapipe_estimator import TestMediaPipePoseEstimator
        import unittest

        # Create a test suite with all tests from TestMediaPipePoseEstimator
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMediaPipePoseEstimator)

        # Run the tests and collect results
        result = unittest.TextTestRunner().run(suite)

        # Check if any tests failed
        assert result.wasSuccessful(), f"MediaPipe estimator tests failed: {result.failures}"

        print("MediaPipe estimator tests passed successfully")