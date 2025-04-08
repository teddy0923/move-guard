# feature_extractors/base_extractor.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction from pose landmarks"""

    def __init__(self, config, movement_type):
        """
        Initialize feature extractor with configuration

        Args:
            config: Configuration dictionary with feature extraction parameters
            movement_type: Type of movement to analyze (e.g., "squat", "ybt")
        """
        self.config = config
        self.movement_type = movement_type
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the feature extractor with movement-specific parameters"""
        pass

    @abstractmethod
    def extract_features(self, landmarks_sequence):
        """
        Extract features from a sequence of landmarks

        Args:
            landmarks_sequence: Array of landmarks for each frame

        Returns:
            features: DataFrame with extracted features
        """
        pass

    @abstractmethod
    def detect_movement_phases(self, landmarks_sequence):
        """
        Detect phases of the movement (e.g., descent, bottom position, ascent)

        Args:
            landmarks_sequence: Array of landmarks for each frame

        Returns:
            phases: Dictionary with frame indices for each movement phase
        """
        pass

    @abstractmethod
    def calculate_angles(self, landmarks, angle_config):
        """
        Calculate joint angles from landmarks

        Args:
            landmarks: Dictionary of landmarks with their 3D coordinates
            angle_config: Configuration for angle calculation

        Returns:
            angle: Calculated angle in degrees
        """
        pass

    @abstractmethod
    def calculate_distances(self, landmarks, distance_config):
        """
        Calculate distances between landmarks

        Args:
            landmarks: Dictionary of landmarks with their 3D coordinates
            distance_config: Configuration for distance calculation

        Returns:
            distance: Calculated distance
        """
        pass