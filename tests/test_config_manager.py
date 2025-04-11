# tests/test_config_manager.py
import os
import sys
import pytest
from pathlib import Path

# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from core.config_manager import ConfigLoader


class TestConfigManager:
    """Tests for the ConfigLoader class"""

    def test_load_config(self):
        """Test loading a configuration file"""
        # Initialize config loader
        loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))

        # Load default configuration
        config = loader.load_config('default')

        # Verify config is loaded correctly
        assert config is not None
        assert 'paths' in config
        assert 'data' in config['paths']
        assert 'landmarks' in config['paths']['data']

        # Verify pose estimation configuration
        assert 'pose_estimation' in config
        assert 'algorithm' in config['pose_estimation']
        assert config['pose_estimation']['algorithm'] == 'mediapipe'

        # Verify MediaPipe-specific configuration
        assert 'mediapipe' in config['pose_estimation']
        assert 'model_complexity' in config['pose_estimation']['mediapipe']

    def test_load_movement_config(self):
        """Test loading a movement-specific configuration file"""
        # Initialize config loader
        loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))

        # Load squat configuration
        config = loader.load_config('squat')

        # Verify config is loaded correctly
        assert config is not None
        assert 'phases' in config
        assert 'landmarks' in config
        assert 'features' in config

        # Verify feature types
        assert 'angles' in config['features']
        assert 'distances' in config['features']
        assert 'ratios' in config['features']

        # Check specific feature definitions
        assert 'femoral_angle' in config['features']['angles']
        assert 'wrist_to_foot' in config['features']['distances']
        assert 'knee_ankle_separation' in config['features']['ratios']

    def test_merge_configs(self):
        """Test merging configurations"""
        # Initialize config loader
        loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))

        # Load and merge configurations
        merged_config = loader.merge_configs('default', 'squat')

        # Verify merged config contains elements from both
        assert merged_config is not None
        assert 'pose_estimation' in merged_config  # From default
        assert 'features' in merged_config  # From squat

        # Verify default paths are preserved
        assert 'paths' in merged_config
        assert 'data' in merged_config['paths']

        # Verify squat features are added
        assert 'angles' in merged_config['features']
        assert 'femoral_angle' in merged_config['features']['angles']

    def test_nonexistent_config(self):
        """Test loading a non-existent configuration file"""
        # Initialize config loader
        loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))

        # Attempt to load non-existent configuration
        config = loader.load_config('nonexistent')

        # Verify result is None
        assert config is None