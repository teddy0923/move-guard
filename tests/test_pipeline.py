# tests/test_pipeline.py
import os
import sys
import pytest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import src.pose_estimators.mediapipe_estimator

# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigLoader
from src.core.pipeline import Pipeline




class TestPipeline:
    """Tests for the Pipeline orchestrator"""

    @pytest.fixture
    def config(self):
        """Load the test configuration"""
        loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))
        config = loader.merge_configs('default', 'squat')
        return config

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_pipeline_initialization(self, config):
        """Test pipeline initialization with components"""
        # Initialize pipeline
        pipeline = Pipeline(config)

        # Verify components are initialized
        assert pipeline.components is not None
        assert 'pose_estimator' in pipeline.components
        assert 'feature_extractor' in pipeline.components
        assert 'ml_model' in pipeline.components

        # Verify component types
        from src.pose_estimators.mediapipe_estimator import MediaPipePoseEstimator
        from src.feature_extractors.squat_feature_extractor import SquatFeatureExtractor
        from src.models.traditional.random_forest_model import RandomForestModel

        assert isinstance(pipeline.components['pose_estimator'], MediaPipePoseEstimator)
        assert isinstance(pipeline.components['feature_extractor'], SquatFeatureExtractor)
        assert isinstance(pipeline.components['ml_model'], RandomForestModel)

    def test_pipeline_component_access(self, config):
        """Test accessing pipeline components"""
        # Initialize pipeline
        pipeline = Pipeline(config)

        # Access components
        pose_estimator = pipeline.get_component('pose_estimator')
        feature_extractor = pipeline.get_component('feature_extractor')
        ml_model = pipeline.get_component('ml_model')

        # Verify components are accessible
        assert pose_estimator is not None
        assert feature_extractor is not None
        assert ml_model is not None

        # Verify non-existent component returns None
        assert pipeline.get_component('nonexistent') is None

    def test_pipeline_process_video(self, config, temp_dir, monkeypatch):
        """Test pipeline video processing flow"""
        # Create a simple test video file
        test_video = os.path.join(temp_dir, 'test_video.mp4')
        with open(test_video, 'w') as f:
            f.write('dummy video content')

        # Create output directory
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Mock the video processing method to avoid actual video processing
        def mock_process_video(self, video_path, output_path, video_segment=None):
            # Return dummy landmarks array
            return np.random.random((10, 33, 3))

        # Apply mock
        import src.pose_estimators.mediapipe_estimator
        monkeypatch.setattr(src.pose_estimators.mediapipe_estimator.MediaPipePoseEstimator,
                            'process_video', mock_process_video)

        # Initialize pipeline
        pipeline = Pipeline(config)

        # Process video
        result = pipeline.process_video(test_video, output_dir)

        # Verify successful processing
        assert result['success'] is True
        assert 'landmarks' in result['results']
        assert 'features' in result['results']

        # Verify landmarks
        landmarks = result['results']['landmarks']
        assert landmarks is not None
        assert landmarks.shape == (10, 33, 3)

        # Verify features (structure depends on feature extractor implementation)
        features = result['results']['features']
        assert features is not None
        assert isinstance(features, pd.DataFrame)

    def test_pipeline_train_model(self, config, monkeypatch):
        """Test pipeline model training flow"""
        # Create dummy features and labels
        features = pd.DataFrame({
            'feature1': np.random.random(10),
            'feature2': np.random.random(10),
            'feature3': np.random.random(10)
        })

        labels = np.array([1, 2, 1, 3, 2, 1, 2, 3, 2, 1])

        # Initialize pipeline
        pipeline = Pipeline(config)

        # Train model
        result = pipeline.train_model(features, labels)

        # Verify training result
        assert result['success'] is True
        assert 'training_metrics' in result

        # Verify metrics (structure depends on ML model implementation)
        metrics = result['training_metrics']
        assert metrics is not None
        assert 'accuracy' in metrics

    def test_pipeline_evaluate_model(self, config, monkeypatch):
        """Test pipeline model evaluation flow"""
        # Create dummy features and labels
        features = pd.DataFrame({
            'feature1': np.random.random(10),
            'feature2': np.random.random(10),
            'feature3': np.random.random(10)
        })

        labels = np.array([1, 2, 1, 3, 2, 1, 2, 3, 2, 1])

        # Initialize pipeline
        pipeline = Pipeline(config)

        # Ensure model is trained first
        pipeline.train_model(features, labels)

        # Evaluate model
        result = pipeline.evaluate_model(features, labels)

        # Verify evaluation result
        assert result['success'] is True
        assert 'evaluation_metrics' in result

        # Verify metrics (structure depends on ML model implementation)
        metrics = result['evaluation_metrics']
        assert metrics is not None
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics

    def test_pipeline_error_handling(self, config, temp_dir, monkeypatch):
        """Test pipeline error handling"""
        # Create a non-existent test video path
        test_video = os.path.join(temp_dir, 'nonexistent_video.mp4')

        # Initialize pipeline
        pipeline = Pipeline(config)

        # Process video with non-existent file
        result = pipeline.process_video(test_video, temp_dir)

        # Verify error handling
        assert result['success'] is False
        assert 'error' in result