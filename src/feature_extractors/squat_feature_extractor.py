# feature_extractors/squat_feature_extractor.py
import numpy as np
import pandas as pd
import logging
import math
from src.feature_extractors.base_extractor import BaseFeatureExtractor


class SquatFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for squat movements, extracting biomechanical features
    that relate to movement quality and injury risk.
    """

    def _initialize(self):
        """Initialize the squat feature extractor with movement-specific parameters"""
        logging.info("Initializing SquatFeatureExtractor")

        # Get squat-specific configuration
        self.features_config = self.config.get(self.movement_type, {}).get('features', {})
        self.landmarks_config = self.config.get(self.movement_type, {}).get('landmarks', {})

        # Log configuration to verify loading
        feature_types = list(self.features_config.keys())
        logging.info(f"Loaded feature types for {self.movement_type}: {feature_types}")

        # Extract feature names for logging
        all_features = []
        for feature_type, features in self.features_config.items():
            all_features.extend(list(features.keys()))

        logging.info(f"Total features defined: {len(all_features)}")

        # Initialize landmark indices dictionary for quick access
        self.landmark_indices = {
            # MediaPipe landmark indices
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }

    def extract_features(self, landmarks_sequence, video_metadata=None):
        """
        Extract features from a sequence of landmarks

        Args:
            landmarks_sequence: Array of landmarks for each frame
            video_metadata: Optional dictionary with video metadata

        Returns:
            features: DataFrame with extracted features
        """
        logging.info(f"Extracting features from landmarks with shape {landmarks_sequence.shape}")

        # Initialize features dictionary
        features = {}

        # Get video ID from metadata if available
        video_id = "unknown"
        view_angle = "front"  # Default view angle

        if video_metadata:
            if 'video_id' in video_metadata:
                video_id = video_metadata['video_id']
            elif 'file_name' in video_metadata:
                video_id = video_metadata['file_name']

            # Get view angle from metadata
            if 'View_Angle' in video_metadata:
                view_angle = video_metadata['View_Angle']
            elif 'view_angle' in video_metadata:
                view_angle = video_metadata['view_angle']

        features['video_id'] = video_id

        # Detect movement phases
        phases = self.detect_movement_phases(landmarks_sequence)

        # Calculate ankle angle for each frame
        ankle_angles = []
        for frame_idx, landmarks in enumerate(landmarks_sequence):
            try:
                angle = self.calculate_ankle_angle(landmarks, view_angle)
                ankle_angles.append(angle)
            except Exception as e:
                logging.error(f"Error calculating ankle angle for frame {frame_idx}: {str(e)}")
                ankle_angles.append(0.0)  # Use default value on error

        # Calculate aggregate ankle angle statistics
        if ankle_angles:
            features['ankle_angle_min'] = min(ankle_angles)
            features['ankle_angle_max'] = max(ankle_angles)
            features['ankle_angle_mean'] = sum(ankle_angles) / len(ankle_angles)
            features['frame_angles'] = ankle_angles  # Store all angles for detailed output
        else:
            features['ankle_angle_min'] = 0.0
            features['ankle_angle_max'] = 0.0
            features['ankle_angle_mean'] = 0.0
            features['frame_angles'] = []

        # Return as DataFrame
        features_df = pd.DataFrame([features])
        logging.info(f"Generated {features_df.shape[1]} features")

        return features_df

    def detect_movement_phases(self, landmarks_sequence):
        """
        Detect phases of the squat movement

        Args:
            landmarks_sequence: Array of landmarks for each frame

        Returns:
            phases: Dictionary with frame indices for each movement phase
        """
        # For now, create simplified phases based on frames
        num_frames = len(landmarks_sequence)

        phases = {
            'start_position': [0, int(num_frames * 0.1)],
            'descent': [int(num_frames * 0.1), int(num_frames * 0.4)],
            'bottom_position': [int(num_frames * 0.4), int(num_frames * 0.6)],
            'ascent': [int(num_frames * 0.6), int(num_frames * 0.9)],
            'end_position': [int(num_frames * 0.9), num_frames - 1]
        }

        logging.info(f"Detected movement phases: {phases}")
        return phases

    def calculate_ankle_angle(self, landmarks, view_angle):
        """
        Calculate ankle angle from landmarks

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated ankle angle in degrees
        """
        # Determine which side to use based on view_angle
        if view_angle == "sag_left":
            heel_idx = self.landmark_indices['left_heel']
            foot_idx = self.landmark_indices['left_foot_index']
            ankle_idx = self.landmark_indices['left_ankle']
            knee_idx = self.landmark_indices['left_knee']
        elif view_angle == "sag_right":
            heel_idx = self.landmark_indices['right_heel']
            foot_idx = self.landmark_indices['right_foot_index']
            ankle_idx = self.landmark_indices['right_ankle']
            knee_idx = self.landmark_indices['right_knee']
        else:
            # Default to left side
            heel_idx = self.landmark_indices['left_heel']
            foot_idx = self.landmark_indices['left_foot_index']
            ankle_idx = self.landmark_indices['left_ankle']
            knee_idx = self.landmark_indices['left_knee']
            logging.debug(f"Using left side for ankle angle calculation with view angle: {view_angle}")

        # Extract coordinates (only x and y for 2D analysis)
        heel = landmarks[heel_idx][:2]  # Only X and Y
        foot = landmarks[foot_idx][:2]
        ankle = landmarks[ankle_idx][:2]
        knee = landmarks[knee_idx][:2]

        # Calculate vectors
        heel_foot_vector = [foot[0] - heel[0], foot[1] - heel[1]]
        ankle_knee_vector = [knee[0] - ankle[0], knee[1] - ankle[1]]

        # Calculate angle using dot product
        heel_foot_mag = np.sqrt(heel_foot_vector[0] ** 2 + heel_foot_vector[1] ** 2)
        ankle_knee_mag = np.sqrt(ankle_knee_vector[0] ** 2 + ankle_knee_vector[1] ** 2)

        if heel_foot_mag > 0 and ankle_knee_mag > 0:
            dot_product = heel_foot_vector[0] * ankle_knee_vector[0] + heel_foot_vector[1] * ankle_knee_vector[1]
            cos_angle = max(min(dot_product / (heel_foot_mag * ankle_knee_mag), 1.0), -1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            return angle_deg
        else:
            logging.warning("Zero magnitude vector detected when calculating ankle angle")
            return 0

    def calculate_angles(self, landmarks, angle_config):
        """
        Calculate joint angles from landmarks

        Args:
            landmarks: Dictionary of landmarks with their 3D coordinates
            angle_config: Configuration for angle calculation

        Returns:
            angle: Calculated angle in degrees
        """
        # Get landmarks for the three points defining the angle
        points = angle_config.get('points', [])
        if len(points) != 3:
            return 0

        p1 = landmarks[self.landmark_indices[points[0]]][:3]
        p2 = landmarks[self.landmark_indices[points[1]]][:3]
        p3 = landmarks[self.landmark_indices[points[2]]][:3]

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        dot_product = np.dot(v1, v2)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        if v1_mag * v2_mag < 1e-10:  # Check for zero vectors
            return 0

        cos_angle = min(max(dot_product / (v1_mag * v2_mag), -1.0), 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_distances(self, landmarks, distance_config):
        """
        Calculate distances between landmarks

        Args:
            landmarks: Dictionary of landmarks with their 3D coordinates
            distance_config: Configuration for distance calculation

        Returns:
            distance: Calculated distance
        """
        points = distance_config.get('points', [])
        if len(points) != 2:
            return 0

        p1 = landmarks[self.landmark_indices[points[0]]][:3]
        p2 = landmarks[self.landmark_indices[points[1]]][:3]

        # Calculate Euclidean distance
        distance = np.linalg.norm(p1 - p2)

        return distance