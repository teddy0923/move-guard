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
        features['view_angle'] = view_angle

        # Detect movement phases
        phases = self.detect_movement_phases(landmarks_sequence)

        # Only calculate angles for sagittal views
        if view_angle.lower() in ["sag_left", "sag_right", "sagittal_left", "sagittal_right", "sagittal"]:
            # Calculate ankle angle for each frame
            ankle_angles = []
            knee_angles = []  # New array for knee angles
            hip_angles = []  # New array for hip angles

            for frame_idx, landmarks in enumerate(landmarks_sequence):
                try:
                    # Calculate ankle angle
                    ankle_angle = self.calculate_ankle_angle(landmarks, view_angle)
                    ankle_angles.append(ankle_angle)

                    # Calculate knee angle
                    knee_angle = self.calculate_knee_angle(landmarks, view_angle)
                    knee_angles.append(knee_angle)

                    # Calculate hip angle
                    hip_angle = self.calculate_hip_flexion(landmarks, view_angle)
                    hip_angles.append(hip_angle)
                except Exception as e:
                    logging.error(f"Error calculating angles for frame {frame_idx}: {str(e)}")
                    ankle_angles.append(0.0)  # Use default value on error
                    knee_angles.append(0.0)  # Use default value on error
                    hip_angles.append(0.0)  # Use default value on error

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

            # Calculate aggregate knee angle statistics
            if knee_angles:
                features['knee_angle_min'] = min(knee_angles)
                features['knee_angle_max'] = max(knee_angles)
                features['knee_angle_mean'] = sum(knee_angles) / len(knee_angles)
                features['knee_frame_angles'] = knee_angles  # Store all knee angles for detailed output
            else:
                features['knee_angle_min'] = 0.0
                features['knee_angle_max'] = 0.0
                features['knee_angle_mean'] = 0.0
                features['knee_frame_angles'] = []

            # Calculate aggregate hip angle statistics
            if hip_angles:
                features['hip_flexion_min'] = min(hip_angles)
                features['hip_flexion_max'] = max(hip_angles)
                features['hip_flexion_mean'] = sum(hip_angles) / len(hip_angles)
                features['hip_flexion_frame_angles'] = hip_angles  # Store all hip angles for detailed output
            else:
                features['hip_flexion_min'] = 0.0
                features['hip_flexion_max'] = 0.0
                features['hip_flexion_mean'] = 0.0
                features['hip_flexion_frame_angles'] = []
        else:
            logging.info(f"Skipping angle calculation for non-sagittal view: {view_angle}")
            # Add empty angle fields to maintain consistent DataFrame structure
            features['ankle_angle_min'] = None
            features['ankle_angle_max'] = None
            features['ankle_angle_mean'] = None
            features['frame_angles'] = []
            features['knee_angle_min'] = None
            features['knee_angle_max'] = None
            features['knee_angle_mean'] = None
            features['knee_frame_angles'] = []
            features['hip_flexion_min'] = None
            features['hip_flexion_max'] = None
            features['hip_flexion_mean'] = None
            features['hip_flexion_frame_angles'] = []

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
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            heel_idx = self.landmark_indices['left_heel']
            foot_idx = self.landmark_indices['left_foot_index']
            ankle_idx = self.landmark_indices['left_ankle']
            knee_idx = self.landmark_indices['left_knee']
        else:  # Default to right side for sagittal_right
            heel_idx = self.landmark_indices['right_heel']
            foot_idx = self.landmark_indices['right_foot_index']
            ankle_idx = self.landmark_indices['right_ankle']
            knee_idx = self.landmark_indices['right_knee']

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

    def calculate_knee_angle(self, landmarks, view_angle):
        """
        Calculate knee angle from landmarks

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated knee angle in degrees
        """
        # Determine which side to use based on view_angle
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            hip_idx = self.landmark_indices['left_hip']
            knee_idx = self.landmark_indices['left_knee']
            ankle_idx = self.landmark_indices['left_ankle']
        else:  # Default to right side for sagittal_right
            hip_idx = self.landmark_indices['right_hip']
            knee_idx = self.landmark_indices['right_knee']
            ankle_idx = self.landmark_indices['right_ankle']

        # Extract coordinates (only x and y for 2D analysis)
        hip = landmarks[hip_idx][:2]  # Only X and Y
        knee = landmarks[knee_idx][:2]
        ankle = landmarks[ankle_idx][:2]

        # Calculate vectors
        knee_hip_vector = [hip[0] - knee[0], hip[1] - knee[1]]
        knee_ankle_vector = [ankle[0] - knee[0], ankle[1] - knee[1]]

        # Calculate angle using dot product
        knee_hip_mag = np.sqrt(knee_hip_vector[0] ** 2 + knee_hip_vector[1] ** 2)
        knee_ankle_mag = np.sqrt(knee_ankle_vector[0] ** 2 + knee_ankle_vector[1] ** 2)

        if knee_hip_mag > 0 and knee_ankle_mag > 0:
            dot_product = knee_hip_vector[0] * knee_ankle_vector[0] + knee_hip_vector[1] * knee_ankle_vector[1]
            cos_angle = max(min(dot_product / (knee_hip_mag * knee_ankle_mag), 1.0), -1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            return angle_deg
        else:
            logging.warning("Zero magnitude vector detected when calculating knee angle")
            return 0

    def calculate_hip_flexion(self, landmarks, view_angle):
        """
        Calculate hip flexion angle from landmarks (angle between shoulder, hip and knee)

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated hip flexion angle in degrees
        """
        # Determine which side to use based on view_angle
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            shoulder_idx = self.landmark_indices['left_shoulder']
            hip_idx = self.landmark_indices['left_hip']
            knee_idx = self.landmark_indices['left_knee']
        else:  # Default to right side for sagittal_right
            shoulder_idx = self.landmark_indices['right_shoulder']
            hip_idx = self.landmark_indices['right_hip']
            knee_idx = self.landmark_indices['right_knee']

        # Extract coordinates (only x and y for 2D analysis)
        shoulder = landmarks[shoulder_idx][:2]  # Only X and Y
        hip = landmarks[hip_idx][:2]
        knee = landmarks[knee_idx][:2]

        # Calculate vectors
        hip_shoulder_vector = [shoulder[0] - hip[0], shoulder[1] - hip[1]]
        hip_knee_vector = [knee[0] - hip[0], knee[1] - hip[1]]

        # Calculate angle using dot product
        hip_shoulder_mag = np.sqrt(hip_shoulder_vector[0] ** 2 + hip_shoulder_vector[1] ** 2)
        hip_knee_mag = np.sqrt(hip_knee_vector[0] ** 2 + hip_knee_vector[1] ** 2)

        if hip_shoulder_mag > 0 and hip_knee_mag > 0:
            dot_product = hip_shoulder_vector[0] * hip_knee_vector[0] + hip_shoulder_vector[1] * hip_knee_vector[1]
            cos_angle = max(min(dot_product / (hip_shoulder_mag * hip_knee_mag), 1.0), -1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            return angle_deg
        else:
            logging.warning("Zero magnitude vector detected when calculating hip flexion angle")
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