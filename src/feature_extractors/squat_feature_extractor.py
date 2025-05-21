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

        # Calculate angles based on view
        if view_angle.lower() in ["sag_left", "sag_right", "sagittal_left", "sagittal_right", "sagittal"]:
            # Initialize arrays for storing angles
            ankle_angles = []  # New array for ankle angles
            knee_angles = []  # New array for knee angles
            hip_angles = []  # New array for hip angles
            shoulder_angles = []  # New array for shoulder angles
            femoral_angles = []  # New array for femoral angles
            trunk_tibia_angles = []  # New array for trunk-tibia angles

            # Process each frame
            for frame_idx, landmarks in enumerate(landmarks_sequence):
                try:
                    # Calculate angles for sagittal view
                    ankle_angle = self.calculate_ankle_angle(landmarks, view_angle)
                    knee_angle = self.calculate_knee_angle(landmarks, view_angle)
                    hip_angle = self.calculate_hip_flexion(landmarks, view_angle)
                    shoulder_angle = self.calculate_shoulder_flexion(landmarks, view_angle)
                    femoral_angle = self.calculate_femoral_angle(landmarks, view_angle)
                    trunk_tibia_angle = self.calculate_trunk_tibia_angle(landmarks, view_angle)

                    # Store angles
                    ankle_angles.append(ankle_angle)
                    knee_angles.append(knee_angle)
                    hip_angles.append(hip_angle)
                    shoulder_angles.append(shoulder_angle)
                    femoral_angles.append(femoral_angle)
                    trunk_tibia_angles.append(trunk_tibia_angle)

                except Exception as e:
                    logging.error(f"Error calculating angles for frame {frame_idx}: {str(e)}")
                    ankle_angles.append(0.0)
                    knee_angles.append(0.0)
                    hip_angles.append(0.0)
                    shoulder_angles.append(0.0)
                    femoral_angles.append(0.0)
                    trunk_tibia_angles.append(0.0)

            # Calculate aggregate ankle angle statistics
            if ankle_angles:
                features['ankle_angle_min'] = min(ankle_angles)
                features['ankle_angle_max'] = max(ankle_angles)
                features['ankle_angle_mean'] = sum(ankle_angles) / len(ankle_angles)
                features['ankle_frame_angles'] = ankle_angles
            else:
                features['ankle_angle_min'] = 0.0
                features['ankle_angle_max'] = 0.0
                features['ankle_angle_mean'] = 0.0
                features['ankle_frame_angles'] = []

            # Calculate aggregate knee angle statistics
            if knee_angles:
                features['knee_angle_min'] = min(knee_angles)
                features['knee_angle_max'] = max(knee_angles)
                features['knee_angle_mean'] = sum(knee_angles) / len(knee_angles)
                features['knee_frame_angles'] = knee_angles
            else:
                features['knee_angle_min'] = 0.0
                features['knee_angle_max'] = 0.0
                features['knee_angle_mean'] = 0.0
                features['knee_frame_angles'] = []

            # Calculate aggregate hip flexion statistics
            if hip_angles:
                features['hip_flexion_min'] = min(hip_angles)
                features['hip_flexion_max'] = max(hip_angles)
                features['hip_flexion_mean'] = sum(hip_angles) / len(hip_angles)
                features['hip_flexion_frame_angles'] = hip_angles
            else:
                features['hip_flexion_min'] = 0.0
                features['hip_flexion_max'] = 0.0
                features['hip_flexion_mean'] = 0.0
                features['hip_flexion_frame_angles'] = []

            # Calculate aggregate shoulder flexion statistics
            if shoulder_angles:
                features['shoulder_flexion_min'] = min(shoulder_angles)
                features['shoulder_flexion_max'] = max(shoulder_angles)
                features['shoulder_flexion_mean'] = sum(shoulder_angles) / len(shoulder_angles)
                features['shoulder_flexion_frame_angles'] = shoulder_angles
            else:
                features['shoulder_flexion_min'] = 0.0
                features['shoulder_flexion_max'] = 0.0
                features['shoulder_flexion_mean'] = 0.0
                features['shoulder_flexion_frame_angles'] = []

            # Calculate aggregate femoral angle statistics
            if femoral_angles:
                features['femoral_angle_min'] = min(femoral_angles)
                features['femoral_angle_max'] = max(femoral_angles)
                features['femoral_angle_mean'] = sum(femoral_angles) / len(femoral_angles)
                features['femoral_frame_angles'] = femoral_angles
            else:
                features['femoral_angle_min'] = 0.0
                features['femoral_angle_max'] = 0.0
                features['femoral_angle_mean'] = 0.0
                features['femoral_frame_angles'] = []

            # Calculate aggregate trunk-tibia angle statistics
            if trunk_tibia_angles:
                features['trunk_tibia_angle_min'] = min(trunk_tibia_angles)
                features['trunk_tibia_angle_max'] = max(trunk_tibia_angles)
                features['trunk_tibia_angle_mean'] = sum(trunk_tibia_angles) / len(trunk_tibia_angles)
                features['trunk_tibia_angle_frame_angles'] = trunk_tibia_angles
            else:
                features['trunk_tibia_angle_min'] = 0.0
                features['trunk_tibia_angle_max'] = 0.0
                features['trunk_tibia_angle_mean'] = 0.0
                features['trunk_tibia_angle_frame_angles'] = []

        elif view_angle.lower() == "front":
            features_df = self.extract_front_features(landmarks_sequence, video_metadata)
            features = features_df.to_dict('records')[0]

        else:
            logging.info(f"Skipping angle calculation for non-sagittal view: {view_angle}")
            # Add empty angle fields to maintain consistent DataFrame structure
            features['ankle_angle_min'] = None
            features['ankle_angle_max'] = None
            features['ankle_angle_mean'] = None
            features['ankle_frame_angles'] = []
            features['knee_angle_min'] = None
            features['knee_angle_max'] = None
            features['knee_angle_mean'] = None
            features['knee_frame_angles'] = []
            features['hip_flexion_min'] = None
            features['hip_flexion_max'] = None
            features['hip_flexion_mean'] = None
            features['hip_flexion_frame_angles'] = []
            features['shoulder_flexion_min'] = None
            features['shoulder_flexion_max'] = None
            features['shoulder_flexion_mean'] = None
            features['shoulder_flexion_frame_angles'] = []
            features['femoral_angle_min'] = None
            features['femoral_angle_max'] = None
            features['femoral_angle_mean'] = None
            features['femoral_frame_angles'] = []
            features['trunk_tibia_angle_min'] = None
            features['trunk_tibia_angle_max'] = None
            features['trunk_tibia_angle_mean'] = None
            features['trunk_tibia_angle_frame_angles'] = []

        # Return as DataFrame
        features_df = pd.DataFrame([features])
        logging.info(f"Generated {features_df.shape[1]} features")

        return features_df

    def extract_front_features(self, landmarks_sequence, video_metadata=None):
        """Extract front view features from a sequence of landmarks.
        
        Args:
            landmarks_sequence: Array of landmarks for each frame
            video_metadata: Optional dictionary with video metadata
            
        Returns:
            features_df: DataFrame with extracted front view features per frame
        """
        # Lists to store per-frame data
        frame_data = []
        
        # Get video ID from metadata
        video_id = video_metadata.get('video_id', 'unknown') if video_metadata else 'unknown'
        
        # Process each frame
        for frame_idx, landmarks in enumerate(landmarks_sequence):
            try:
                # Calculate angles for front view
                left_elbow = self.calculate_elbow_flexion_angle(landmarks, 'left')
                right_elbow = self.calculate_elbow_flexion_angle(landmarks, 'right')
                hip_alignment = self.calculate_hip_alignment_front(landmarks)
                shoulder_alignment = self.calculate_shoulder_alignment_front(landmarks)
                knee_dist, ankle_dist, knee_ankle_ratio = self.calculate_knee_ankle_distances_front(landmarks)
                
                # Create frame entry
                frame_entry = {
                    'frame': frame_idx,
                    'video_id': video_id,
                    'left_elbow_angle': left_elbow,
                    'right_elbow_angle': right_elbow,
                    'hip_alignment': hip_alignment,
                    'shoulder_alignment': shoulder_alignment,
                    'knee_distance': knee_dist,
                    'ankle_distance': ankle_dist,
                    'knee_ankle_ratio': knee_ankle_ratio
                }
                frame_data.append(frame_entry)
                
            except Exception as e:
                logging.error(f"Error calculating front view angles for frame {frame_idx}: {str(e)}")
                frame_data.append({
                    'frame': frame_idx,
                    'video_id': video_id,
                    'left_elbow_angle': 0.0,
                    'right_elbow_angle': 0.0,
                    'hip_alignment': 0.0,
                    'shoulder_alignment': 0.0,
                    'knee_distance': 0.0,
                    'ankle_distance': 0.0,
                    'knee_ankle_ratio': 0.0
                })
        
        # Create DataFrame with all frames
        features_df = pd.DataFrame(frame_data)
        logging.info(f"Generated frame-by-frame features for {len(frame_data)} frames")
        
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

    def calculate_shoulder_flexion(self, landmarks, view_angle):
        """
        Calculate shoulder flexion angle from landmarks (angle between hip, shoulder and elbow)

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated shoulder flexion angle in degrees
        """
        # Determine which side to use based on view_angle
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            hip_idx = self.landmark_indices['left_hip']
            shoulder_idx = self.landmark_indices['left_shoulder']
            elbow_idx = self.landmark_indices['left_elbow']
        else:  # Default to right side for sagittal_right
            hip_idx = self.landmark_indices['right_hip']
            shoulder_idx = self.landmark_indices['right_shoulder']
            elbow_idx = self.landmark_indices['right_elbow']

        # Extract coordinates (only x and y for 2D analysis)
        hip = landmarks[hip_idx][:2]  # Only X and Y
        shoulder = landmarks[shoulder_idx][:2]
        elbow = landmarks[elbow_idx][:2]

        # Calculate vectors
        shoulder_hip_vector = [hip[0] - shoulder[0], hip[1] - shoulder[1]]
        shoulder_elbow_vector = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]

        # Calculate angle using dot product
        shoulder_hip_mag = np.sqrt(shoulder_hip_vector[0] ** 2 + shoulder_hip_vector[1] ** 2)
        shoulder_elbow_mag = np.sqrt(shoulder_elbow_vector[0] ** 2 + shoulder_elbow_vector[1] ** 2)

        if shoulder_hip_mag > 0 and shoulder_elbow_mag > 0:
            dot_product = shoulder_hip_vector[0] * shoulder_elbow_vector[0] + shoulder_hip_vector[1] * \
                          shoulder_elbow_vector[1]
            cos_angle = max(min(dot_product / (shoulder_hip_mag * shoulder_elbow_mag), 1.0), -1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            return angle_deg
        else:
            logging.warning("Zero magnitude vector detected when calculating shoulder flexion angle")
            return 0

    def calculate_femoral_angle(self, landmarks, view_angle):
        """
        Calculate femoral angle (angle between femur segment and horizontal)

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated femoral angle in degrees
            90° = vertical up
            0° = horizontal
            -90° = vertical down
        """
        # Determine which side to use based on view_angle
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            knee_idx = self.landmark_indices['left_knee']
            hip_idx = self.landmark_indices['left_hip']
        else:  # Default to right side for sagittal_right
            knee_idx = self.landmark_indices['right_knee']
            hip_idx = self.landmark_indices['right_hip']

        # Extract coordinates (only x and y for 2D analysis)
        knee = landmarks[knee_idx][:2]  # Only X and Y
        hip = landmarks[hip_idx][:2]

        # Define vectors
        femur_vector = np.array([hip[0] - knee[0], hip[1] - knee[1]])  # Vector from knee to hip
        horizontal_vector = np.array([1.0, 0.0])  # Fixed horizontal vector (unit vector along x-axis)
        '''
        # Calculate angle using arctan2
        # arctan2(y2-y1, x2-x1) gives angle between vectors
        angle_rad = np.arctan2(femur_vector[1], femur_vector[0]) - np.arctan2(horizontal_vector[1], horizontal_vector[0])

        # Normalize angle to [-pi, pi]
        angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi

        # Convert to degrees and negate to account for image coordinates
        angle_deg = np.degrees(angle_rad)
        return angle_deg
        '''

        # Normalize vectors
        femur_unit = femur_vector / np.linalg.norm(femur_vector)

        # Compute angle via dot product
        dot = np.dot(femur_unit, horizontal_vector)
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        angle_deg = 180 - angle_deg

        return angle_deg  # This gives us:
        # 90° when femur is vertical up
        # 0° when femur is horizontal
        # -90° when femur is vertical down

    def calculate_trunk_tibia_angle(self, landmarks, view_angle):
        """
        Calculate angle between trunk and tibia vectors

        Args:
            landmarks: Array of landmarks for a single frame
            view_angle: Camera view angle (e.g., "sag_left", "sag_right", "front")

        Returns:
            angle: Calculated trunk-tibia angle in degrees
            0° = trunk parallel to tibia
            90° = trunk perpendicular to tibia
        """
        # Determine which side to use based on view_angle
        if view_angle.lower() in ["sag_left", "sagittal_left"]:
            ankle_idx = self.landmark_indices['left_ankle']
            knee_idx = self.landmark_indices['left_knee']
            hip_idx = self.landmark_indices['left_hip']
            shoulder_idx = self.landmark_indices['left_shoulder']
        else:  # Default to right side for sagittal_right
            ankle_idx = self.landmark_indices['right_ankle']
            knee_idx = self.landmark_indices['right_knee']
            hip_idx = self.landmark_indices['right_hip']
            shoulder_idx = self.landmark_indices['right_shoulder']

        # Extract coordinates (only x and y for 2D analysis)
        ankle = landmarks[ankle_idx][:2]
        knee = landmarks[knee_idx][:2]
        hip = landmarks[hip_idx][:2]
        shoulder = landmarks[shoulder_idx][:2]

        # Compute tibia vector (from ankle to knee)
        tibia_vector = np.array([knee[0] - ankle[0], knee[1] - ankle[1]])

        # Compute trunk vector (from hip to shoulder)
        trunk_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])

        # Normalize vectors
        tibia_unit = tibia_vector / np.linalg.norm(tibia_vector)
        trunk_unit = trunk_vector / np.linalg.norm(trunk_vector)

        # Compute perpendicular vector to tibia (rotate 90 degrees)
        tibia_perp = np.array([-tibia_unit[1], tibia_unit[0]])  # Counter-clockwise rotation

        # Compute dot product between trunk and perpendicular tibia vector
        dot = np.dot(trunk_unit, tibia_perp)

        # Get angle using arccos, clip to handle numerical errors
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # The angle will be 0° when trunk is parallel to tibia
        # and 90° when trunk is perpendicular to tibia
        return angle_deg

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

    def calculate_elbow_flexion_angle(self, landmarks, side):
        """
        Calculate elbow flexion angle from landmarks

        Args:
            landmarks: Array of landmarks for a single frame
            side: Side of the body (left or right)

        Returns:
            angle: Calculated elbow flexion angle in degrees
        """
        if side == 'left':
            shoulder_idx = self.landmark_indices['left_shoulder']
            elbow_idx = self.landmark_indices['left_elbow']
            wrist_idx = self.landmark_indices['left_wrist']
        else:
            shoulder_idx = self.landmark_indices['right_shoulder']
            elbow_idx = self.landmark_indices['right_elbow']
            wrist_idx = self.landmark_indices['right_wrist']

        # Extract coordinates (only x and y for 2D analysis)
        shoulder = landmarks[shoulder_idx][:2]  # Only X and Y
        elbow = landmarks[elbow_idx][:2]
        wrist = landmarks[wrist_idx][:2]

        # Calculate vectors
        shoulder_elbow_vector = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
        elbow_wrist_vector = [wrist[0] - elbow[0], wrist[1] - elbow[1]]

        # Calculate angle using dot product
        shoulder_elbow_mag = np.sqrt(shoulder_elbow_vector[0] ** 2 + shoulder_elbow_vector[1] ** 2)
        elbow_wrist_mag = np.sqrt(elbow_wrist_vector[0] ** 2 + elbow_wrist_vector[1] ** 2)

        if shoulder_elbow_mag > 0 and elbow_wrist_mag > 0:
            dot_product = shoulder_elbow_vector[0] * elbow_wrist_vector[0] + shoulder_elbow_vector[1] * \
                          elbow_wrist_vector[1]
            cos_angle = max(min(dot_product / (shoulder_elbow_mag * elbow_wrist_mag), 1.0), -1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = angle_rad * 180 / np.pi
            return angle_deg
        else:
            logging.warning("Zero magnitude vector detected when calculating elbow flexion angle")
            return 0

    def calculate_knee_ankle_distances_front(self, landmarks):
        """Calculate knee and ankle distances in frontal plane using Euclidean distance.
        
        Args:
            landmarks: Array of landmarks for a single frame
            
        Returns:
            tuple: (knee_distance, ankle_distance, ratio)
                  knee_distance: Euclidean distance between left and right knees
                  ankle_distance: Euclidean distance between left and right ankles
                  ratio: knee_distance / ankle_distance
        """
        try:
            # Get landmarks
            left_knee_idx = self.landmark_indices['left_knee']
            right_knee_idx = self.landmark_indices['right_knee']
            left_ankle_idx = self.landmark_indices['left_ankle']
            right_ankle_idx = self.landmark_indices['right_ankle']
            
            # Extract X,Y coordinates
            left_knee = np.array([landmarks[left_knee_idx][0], landmarks[left_knee_idx][1]])
            right_knee = np.array([landmarks[right_knee_idx][0], landmarks[right_knee_idx][1]])
            left_ankle = np.array([landmarks[left_ankle_idx][0], landmarks[left_ankle_idx][1]])
            right_ankle = np.array([landmarks[right_ankle_idx][0], landmarks[right_ankle_idx][1]])
            
            # Calculate Euclidean distances
            knee_distance = np.linalg.norm(right_knee - left_knee)
            ankle_distance = np.linalg.norm(right_ankle - left_ankle)
            
            # Calculate ratio (knee distance / ankle distance)
            if ankle_distance > 0:
                ratio = knee_distance / ankle_distance
            else:
                ratio = 1.0  # Default to neutral if ankle distance is 0
                
            return knee_distance, ankle_distance, ratio
            
        except Exception as e:
            logging.error(f"Error calculating knee-ankle distances: {str(e)}")
            return 0.0, 0.0, 0.0

    def calculate_hip_alignment_front(self, landmarks):
        """Calculate hip alignment for front view by comparing left and right hip y positions.
        
        Args:
            landmarks: Array of landmarks for a single frame
            
        Returns:
            float: Ratio of hip y positions.
                  Positive ratio: left hip is higher than right
                  Negative ratio: right hip is higher than left
                  1.0 or -1.0: hips are perfectly aligned
        """
        try:
            # Get hip landmarks
            left_hip_idx = self.landmark_indices['left_hip']
            right_hip_idx = self.landmark_indices['right_hip']
            
            # Extract y coordinates (vertical position)
            left_hip_y = landmarks[left_hip_idx][1]  # y coordinate
            right_hip_y = landmarks[right_hip_idx][1]  # y coordinate
            
            # Calculate ratio with sign indicating which hip is higher
            if abs(left_hip_y - right_hip_y) < 1e-6:  # They are equal (within floating point precision)
                return 1.0
            elif left_hip_y > right_hip_y:
                return left_hip_y / right_hip_y  # Positive ratio (left higher)
            else:
                return -right_hip_y / left_hip_y  # Negative ratio (right higher)
            
        except Exception as e:
            logging.error(f"Error calculating hip alignment: {str(e)}")
            return 0.0

    def calculate_shoulder_alignment_front(self, landmarks):
        """Calculate shoulder alignment for front view by comparing left and right shoulder y positions.
        
        Args:
            landmarks: Array of landmarks for a single frame
            
        Returns:
            float: Ratio of shoulder y positions.
                  Positive ratio: left shoulder is higher than right
                  Negative ratio: right shoulder is higher than left
                  1.0 or -1.0: shoulders are perfectly aligned
        """
        try:
            # Get shoulder landmarks
            left_shoulder_idx = self.landmark_indices['left_shoulder']
            right_shoulder_idx = self.landmark_indices['right_shoulder']
            
            # Extract y coordinates (vertical position)
            left_shoulder_y = landmarks[left_shoulder_idx][1]  # y coordinate
            right_shoulder_y = landmarks[right_shoulder_idx][1]  # y coordinate
            
            # Calculate ratio with sign indicating which shoulder is higher
            if abs(left_shoulder_y - right_shoulder_y) < 1e-6:  # They are equal (within floating point precision)
                return 1.0
            elif left_shoulder_y > right_shoulder_y:
                return left_shoulder_y / right_shoulder_y  # Positive ratio (left higher)
            else:
                return -right_shoulder_y / left_shoulder_y  # Negative ratio (right higher)
            
        except Exception as e:
            logging.error(f"Error calculating shoulder alignment: {str(e)}")
            return 0.0