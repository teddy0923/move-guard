# tests/conftest.py
import os
import sys
import pytest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigLoader
from src.pose_estimators.mediapipe_estimator import MediaPipePoseEstimator
from src.feature_extractors.squat_feature_extractor import SquatFeatureExtractor
from src.models.traditional.random_forest_model import RandomForestModel


@pytest.fixture(scope="session")
def project_path():
    """Return the project root path"""
    return project_root


@pytest.fixture(scope="session")
def config_loader():
    """Initialize a config loader for tests"""
    return ConfigLoader(config_dir=os.path.join(project_root, 'config'))


@pytest.fixture(scope="session")
def default_config(config_loader):
    """Load the default configuration"""
    return config_loader.load_config('default')


@pytest.fixture(scope="session")
def squat_config(config_loader):
    """Load the squat configuration"""
    return config_loader.load_config('squat')


@pytest.fixture(scope="session")
def merged_config(config_loader):
    """Load the merged configuration"""
    return config_loader.merge_configs('default', 'squat')


@pytest.fixture(scope="function")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories
        for subdir in ['raw', 'landmarks', 'features', 'models', 'results']:
            os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)

        yield temp_dir


@pytest.fixture(scope="function")
def sample_video(test_data_dir):
    """Create a sample video file for testing"""
    video_path = os.path.join(test_data_dir, 'raw', 'sample_squat.mp4')

    # Create a dummy file
    with open(video_path, 'wb') as f:
        f.write(b'dummy video content')

    return video_path


@pytest.fixture(scope="function")
def sample_landmarks(test_data_dir):
    """Create sample landmarks data for testing"""
    landmarks_path = os.path.join(test_data_dir, 'landmarks', 'sample_squat.npy')

    # Create dummy landmarks data (10 frames, 33 landmarks, 3 coordinates)
    landmarks = np.random.random((10, 33, 3))

    # Save to file
    np.save(landmarks_path, landmarks)

    return landmarks_path


@pytest.fixture(scope="function")
def sample_features(test_data_dir):
    """Create sample features data for testing"""
    features_path = os.path.join(test_data_dir, 'features', 'sample_squat.csv')

    # Create dummy features
    features = pd.DataFrame({
        'video_id': ['sample_squat'],
        'femoral_angle_min': [75.5],
        'femoral_angle_max': [120.2],
        'ankle_angle_min': [80.1],
        'ankle_angle_max': [95.8],
        'hip_height_min': [0.25],
        'hip_height_max': [0.65],
        'knee_ankle_separation_min': [0.92],
        'knee_ankle_separation_max': [1.05]
    })

    # Save to file
    features.to_csv(features_path, index=False)

    return features_path


@pytest.fixture(scope="function")
def sample_labels(test_data_dir):
    """Create sample labels data for testing"""
    labels_path = os.path.join(test_data_dir, 'features', 'labels.csv')

    # Create dummy labels
    labels = pd.DataFrame({
        'video_id': ['sample_squat'],
        'quality': [2]  # Moderate quality
    })

    # Save to file
    labels.to_csv(labels_path, index=False)

    return labels_path


@pytest.fixture(scope="function")
def sample_metadata(test_data_dir):
    """Create sample metadata for testing"""
    metadata_path = os.path.join(test_data_dir, 'metadata.csv')

    # Create dummy metadata
    metadata = pd.DataFrame({
        'Filename': ['sample_squat'],
        'Movement': ['squat'],
        'View_Angle': ['frontal'],
        'rep1_start': [10],
        'rep1_end': [50],
        'rep2_start': [60],
        'rep2_end': [100]
    })

    # Save to file
    metadata.to_csv(metadata_path, index=False)

    return metadata_path


@pytest.fixture(scope="function")
def pose_estimator(merged_config):
    """Initialize a pose estimator for testing"""
    return MediaPipePoseEstimator(merged_config.get('pose_estimation', {}))


@pytest.fixture(scope="function")
def feature_extractor(merged_config):
    """Initialize a feature extractor for testing"""
    return SquatFeatureExtractor(merged_config, 'squat')


@pytest.fixture(scope="function")
def ml_model(merged_config):
    """Initialize a machine learning model for testing"""
    return RandomForestModel(merged_config.get('ml_model', {}))


@pytest.fixture(scope="function")
def test_pipeline(merged_config):
    """Initialize the pipeline with test configuration"""
    from core.pipeline import Pipeline
    return Pipeline(merged_config)


@pytest.fixture(scope="function")
def mock_video_processing(monkeypatch):
    """Mock video processing to avoid actual video processing"""

    def mock_process_video(self, video_path, output_path=None, video_segment=None):
        """Mocked version that returns dummy landmarks"""
        # Create dummy landmarks
        landmarks = np.random.random((10, 33, 3))

        # Save if output path provided
        if output_path:
            from core.file_utils import save_landmarks
            filename = os.path.basename(video_path).split('.')[0]
            save_landmarks(landmarks, output_path, filename)

        return landmarks

    # Apply mock to pose estimator
    import pose_estimators.mediapipe_estimator
    monkeypatch.setattr(pose_estimators.mediapipe_estimator.MediaPipePoseEstimator,
                        'process_video', mock_process_video)


@pytest.fixture(scope="function")
def test_environment():
    """Set up environment variables for testing"""
    # Store original environment
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ['TEST_MODE'] = 'True'

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Define a pytest hook to handle logging during tests
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Configure pytest for testing"""
    import logging

    # Set up logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress overly verbose logs during testing
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)