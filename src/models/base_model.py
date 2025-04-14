# src/models/base_model.py
from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    """Abstract base class for machine learning models"""

    def __init__(self, config):
        """
        Initialize ML model with configuration

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the specific machine learning model"""
        pass

    @abstractmethod
    def train(self, features, labels):
        """
        Train the model on extracted features

        Args:
            features: DataFrame with extracted features
            labels: Array of movement quality labels

        Returns:
            training_metrics: Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, features):
        """
        Predict movement quality from features

        Args:
            features: DataFrame with extracted features

        Returns:
            predictions: Array of predicted movement quality labels
        """
        pass

    @abstractmethod
    def evaluate(self, features, true_labels):
        """
        Evaluate model performance

        Args:
            features: DataFrame with extracted features
            true_labels: Array of true movement quality labels

        Returns:
            evaluation_metrics: Dictionary with evaluation metrics
        """
        pass

    @abstractmethod
    def feature_importance(self):
        """
        Get feature importance from the trained model

        Returns:
            importance: Array of feature importance scores
        """
        pass