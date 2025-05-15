# core/pipeline.py
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union


class Pipeline:
    """
    Orchestrates the entire movement analysis pipeline from video to classification.
    Manages the flow of data between different components of the system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.components = {}
        self.results = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initialize pipeline components based on configuration
        """
        # Initialize pose estimator
        pose_algorithm = self.config.get('pose_estimation', {}).get('algorithm', 'mediapipe')
        try:
            if pose_algorithm == 'mediapipe':
                from src.pose_estimators.mediapipe_estimator import MediaPipePoseEstimator
                self.components['pose_estimator'] = MediaPipePoseEstimator(
                    self.config.get('pose_estimation', {})
                )
            elif pose_algorithm == 'openpose':
                from src.pose_estimators.openpose_estimator import OpenPosePoseEstimator
                self.components['pose_estimator'] = OpenPosePoseEstimator(
                    self.config.get('pose_estimation', {})
                )
            # Add more estimators as needed
            else:
                logging.error(f"Unsupported pose estimation algorithm: {pose_algorithm}")
                raise ValueError(f"Unsupported pose estimation algorithm: {pose_algorithm}")
        except ImportError as e:
            logging.error(f"Failed to import pose estimator: {str(e)}")
            raise

        # Initialize feature extractor based on movement type
        movement_type = self.config.get('feature_extraction', {}).get('default_movement', 'squat')
        try:
            if movement_type == 'squat':
                from src.feature_extractors.squat_feature_extractor import SquatFeatureExtractor
                self.components['feature_extractor'] = SquatFeatureExtractor(
                    self.config, movement_type
                )
            elif movement_type == 'ybt':
                from src.feature_extractors.ybt_feature_extractor import YBTFeatureExtractor
                self.components['feature_extractor'] = YBTFeatureExtractor(
                    self.config, movement_type
                )
            # Add more movement types as needed
            else:
                logging.error(f"Unsupported movement type: {movement_type}")
                raise ValueError(f"Unsupported movement type: {movement_type}")
        except ImportError as e:
            logging.error(f"Failed to import feature extractor: {str(e)}")
            raise

        '''# Initialize ML model 
        ml_algorithm = self.config.get('ml_model', {}).get('algorithm', 'random_forest')
        try:
            if ml_algorithm == 'random_forest':
                from src.ml_models.random_forest_model import RandomForestModel
                self.components['ml_model'] = RandomForestModel(
                    self.config.get('ml_model', {})
                )
            elif ml_algorithm == 'svm':
                from src.ml_models.svm_model import SVMModel
                self.components['ml_model'] = SVMModel(
                    self.config.get('ml_model', {})
                )
            elif ml_algorithm == 'neural_network':
                from src.ml_models.neural_network_model import NeuralNetworkModel
                self.components['ml_model'] = NeuralNetworkModel(
                    self.config.get('ml_model', {})
                )
            # Add more ML models as needed
            else:
                logging.error(f"Unsupported ML algorithm: {ml_algorithm}")
                raise ValueError(f"Unsupported ML algorithm: {ml_algorithm}")
        except ImportError as e:
            logging.error(f"Failed to import ML model: {str(e)}")
            raise '''

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                      video_segment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a video through the entire pipeline

        Args:
            video_path: Path to the input video
            output_path: Optional path to save outputs
            video_segment: Optional dictionary with start_frame and end_frame

        Returns:
            Dictionary with processing results
        """
        from src.core.file_utils import ensure_directory_exists

        if output_path is None:
            output_path = os.path.join(
                self.config.get('paths', {}).get('data', {}).get('processed', 'processed/'),
                Path(video_path).stem
            )

        ensure_directory_exists(output_path)

        try:
            # Step 1: Process video to extract landmarks
            if video_segment:
                logging.info(f"Pipeline passing video_segment: {video_segment}")
            logging.info(f"Extracting landmarks from {video_path}")
            landmarks = self.components['pose_estimator'].process_video(
                video_path, output_path, video_segment
            )


            if landmarks is None:
                logging.error(f"Failed to extract landmarks from {video_path}")
                return {'success': False, 'stage': 'landmark_extraction', 'error': 'Landmark extraction failed'}

            self.results['landmarks'] = landmarks


            # Step 2: Extract features from landmarks
            logging.info("Extracting features from landmarks")
            features = self.components['feature_extractor'].extract_features(landmarks)

            if features is None:
                logging.error("Failed to extract features from landmarks")
                return {'success': False, 'stage': 'feature_extraction', 'error': 'Feature extraction failed'}

            self.results['features'] = features

            # Step 3: Predict movement quality using ML model
            # Note: This step assumes the model is already trained
            if hasattr(self.components['ml_model'], 'model') and self.components['ml_model'].model is not None:
                logging.info("Predicting movement quality")
                predictions = self.components['ml_model'].predict(features)

                if predictions is not None:
                    self.results['predictions'] = predictions

            return {
                'success': True,
                'video_path': video_path,
                'output_path': output_path,
                'results': self.results
            }

        except Exception as e:
            logging.error(f"Error in pipeline processing: {str(e)}")
            return {'success': False, 'error': str(e)}

    def train_model(self, features: Any, labels: Any) -> Dict[str, Any]:
        """
        Train the ML model on extracted features

        Args:
            features: Features for training
            labels: Labels for training

        Returns:
            Dictionary with training results
        """
        try:
            logging.info("Training ML model")
            training_metrics = self.components['ml_model'].train(features, labels)

            if training_metrics is None:
                logging.error("Failed to train ML model")
                return {'success': False, 'stage': 'model_training', 'error': 'Model training failed'}

            self.results['training_metrics'] = training_metrics

            return {
                'success': True,
                'model': self.components['ml_model'],
                'training_metrics': training_metrics
            }

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            return {'success': False, 'error': str(e)}

    def evaluate_model(self, features: Any, labels: Any) -> Dict[str, Any]:
        """
        Evaluate the ML model on test data

        Args:
            features: Features for evaluation
            labels: True labels for evaluation

        Returns:
            Dictionary with evaluation results
        """
        try:
            logging.info("Evaluating ML model")
            evaluation_metrics = self.components['ml_model'].evaluate(features, labels)

            if evaluation_metrics is None:
                logging.error("Failed to evaluate ML model")
                return {'success': False, 'stage': 'model_evaluation', 'error': 'Model evaluation failed'}

            self.results['evaluation_metrics'] = evaluation_metrics

            return {
                'success': True,
                'evaluation_metrics': evaluation_metrics
            }

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_component(self, component_name: str) -> Any:
        """
        Get a specific pipeline component

        Args:
            component_name: Name of the component

        Returns:
            The requested component or None if not found
        """
        return self.components.get(component_name)

    def get_results(self) -> Dict[str, Any]:
        """
        Get all pipeline processing results

        Returns:
            Dictionary with all results
        """
        return self.results