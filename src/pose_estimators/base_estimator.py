# pose_estimators/base_estimator.py
from abc import ABC, abstractmethod
import numpy as np


class BasePoseEstimator(ABC):
    """Abstract base class for all pose estimation algorithms"""

    def __init__(self, config):
        """
        Initialize pose estimator with configuration

        Args:
            config: Configuration dictionary with algorithm-specific parameters
        """
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the specific pose estimation algorithm"""
        pass

    @abstractmethod
    def extract_landmarks(self, frame):
        """
        Extract pose landmarks from a single video frame

        Args:
            frame: Video frame (numpy array in BGR format)

        Returns:
            landmarks: Dictionary of landmarks with their 3D coordinates
        """
        pass

    @abstractmethod
    def process_video(self, video_path, output_path=None):
        """
        Process a video file and extract landmarks for all frames

        Args:
            video_path: Path to the video file
            output_path: Optional path to save processed landmarks

        Returns:
            landmarks_sequence: Array of landmarks for each frame
        """
        pass

    @abstractmethod
    def visualize_landmarks(self, frame, landmarks, output_path=None):
        """
        Visualize landmarks on the input frame

        Args:
            frame: Original video frame
            landmarks: Landmarks to visualize
            output_path: Optional path to save visualized frame

        Returns:
            visualization: Frame with landmarks visualization
        """
        pass