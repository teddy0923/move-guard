# pose_estimators/mediapipe_estimator.py
import os
import cv2
import numpy as np
import logging
import mediapipe as mp
import sys
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from .base_estimator import BasePoseEstimator
from src.core.file_utils import ensure_directory_exists, save_landmarks, save_video_metadata

# Get module logger
logger = logging.getLogger(__name__)

class MediaPipePoseEstimator(BasePoseEstimator):
    """Implementation of pose estimation using MediaPipe Pose"""

    def _initialize(self):
        """Initialize the MediaPipe pose detection pipeline"""
        try:
            # Extract configuration parameters
            mp_config = self.config.get('mediapipe', {})
            self.model_complexity = mp_config.get('model_complexity', 1)
            self.min_detection_confidence = mp_config.get('min_detection_confidence', 0.5)
            self.min_tracking_confidence = mp_config.get('min_tracking_confidence', 0.5)
            self.static_image_mode = mp_config.get('static_image_mode', False)

            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            # Create pose detector
            self.pose = self.mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                static_image_mode=self.static_image_mode
            )

            logger.info(f"MediaPipe Pose initialized with complexity={self.model_complexity}, "
                     f"detection_confidence={self.min_detection_confidence}, "
                     f"tracking_confidence={self.min_tracking_confidence}")

        except Exception as e:
            logger.error(f"Error initializing MediaPipe Pose: {str(e)}")
            raise

    def extract_landmarks(self, frame):
        """
        Extract pose landmarks from a single video frame

        Args:
            frame: Video frame (numpy array in BGR format)

        Returns:
            landmarks: Numpy array of landmarks with their 3D coordinates (33 landmarks x 3 dimensions)
                      or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.pose.process(frame_rgb)

            if not results.pose_landmarks:
                return None

            # Convert landmarks to numpy array
            landmarks = np.zeros((33, 4), dtype=np.float32)  # 33 landmarks x (x,y,z,visibility)
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[i, 0] = landmark.x  # x coordinate [0.0, 1.0]
                landmarks[i, 1] = landmark.y  # y coordinate [0.0, 1.0]
                landmarks[i, 2] = landmark.z  # z coordinate (relative depth)
                landmarks[i, 3] = landmark.visibility  # visibility score [0.0, 1.0]

            return landmarks

        except Exception as e:
            logger.error(f"Error extracting landmarks: {str(e)}")
            return None

    def process_video(self, video_path, output_path=None, video_segment=None):
        """
        Process a video file and extract landmarks for all frames

        Args:
            video_path: Path to the video file
            output_path: Optional path to save processed landmarks
            video_segment: Optional dictionary with start_frame and end_frame keys

        Returns:
            landmarks_sequence: Array of landmarks for each frame or None if error
        """

        # Get view angle from video_segment metadata if available
        # At the beginning of the process_video method, add more verbose logging
        if video_segment:
            logging.info(f"MediaPipe estimator received video_segment: {video_segment}")
            # Check if the View_Angle field exists with different possible capitalizations
            view_angle = None
            for key in ['View_Angle', 'view_angle', 'VIEW_ANGLE', 'View Angle']:
                if key in video_segment:
                    view_angle = video_segment[key]
                    logging.info(f"Found view angle with key '{key}': {view_angle}")
                    break

            if view_angle is None:
                logging.warning(f"View angle not found in metadata. Available keys: {list(video_segment.keys())}")
                view_angle = "front"  # Default
        else:
            logging.info("from-mediapipe-No video segment metadata provided")
            view_angle = "front"  # Default

        logging.info(f"Using view angle: {view_angle}")

        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine start and end frames
            start_frame = 0
            end_frame = total_frames - 1

            """ #diasabled to prcess all video
            if video_segment:
                if 'start_frame' in video_segment:
                    start_frame = max(0, int(video_segment['start_frame']))
                if 'end_frame' in video_segment:
                    end_frame = min(total_frames - 1, int(video_segment['end_frame']))
            """
            logger.info(f"Processing video {os.path.basename(video_path)}: "
                     f"frames {start_frame} to {end_frame} (total: {end_frame - start_frame + 1})")

            # Create writer for visualization if output path is provided
            output_video_path = None
            out = None

            if output_path:
                ensure_directory_exists(output_path)
                output_video_path = os.path.join(output_path,
                                             f"{os.path.splitext(os.path.basename(video_path))[0]}_mediapipe_landmarks.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Process frames
            frame_idx = start_frame
            landmarks_sequence = []

            while cap.isOpened() and frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract landmarks
                landmarks = self.extract_landmarks(frame)

                # Add to sequence if landmarks were detected
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
                else:
                    # If no landmarks detected, add placeholder
                    logger.warning(f"No landmarks detected in frame {frame_idx}")
                    # Adding zeros as placeholder to maintain sequence length
                    landmarks_sequence.append(np.zeros((33, 4), dtype=np.float32))

                # Create visualization if output is requested
                if out is not None:
                    # When visualizing landmarks
                    view_angle = "front"  # Default
                    if video_segment and 'View_Angle' in video_segment:
                        view_angle = video_segment['View_Angle']
                    #logging.info(f"Using view angle for visualization: {view_angle}")
                    visualized_frame = self.visualize_landmarks(frame, landmarks, view_angle=view_angle,frame_number=frame_idx)
                    out.write(visualized_frame)

                frame_idx += 1

                # Log progress periodically
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx - start_frame} of {end_frame - start_frame + 1} frames")

            # Release resources
            cap.release()
            if out is not None:
                out.release()

            # Convert landmarks sequence to numpy array
            landmarks_sequence = np.array(landmarks_sequence)

            # Save landmarks if output path is provided
            # Save landmarks if output path is provided
            if output_path:
                # Create metadata
                video_metadata = {
                    'fps': fps,
                    'total_frames': len(landmarks_sequence),
                    'timestamps': [i / fps for i in range(len(landmarks_sequence))],
                    'video_id': os.path.basename(video_path),
                    'processing_date': datetime.now().isoformat(),
                    'view_angle': view_angle if view_angle else 'front',
                    'pose_estimator': 'mediapipe',  # Add this line
                    'pose_estimator_version': mp.__version__,  # Optionally add version information
                    'pose_estimator_config': {  # Include the specific configuration used
                        'model_complexity': self.model_complexity,
                        'min_detection_confidence': self.min_detection_confidence,
                        'min_tracking_confidence': self.min_tracking_confidence,
                        'static_image_mode': self.static_image_mode
                    }
                }

                # Pass to save_landmarks
                save_landmarks(
                    landmarks_sequence,
                    output_path,
                    os.path.splitext(os.path.basename(video_path))[0],
                    estimator_name=self.config.get('algorithm', 'mediapipe'),  # Get from config
                    metadata=video_metadata
                )

                logger.info(
                    f"Landmarks saved to {os.path.join(output_path, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy")}")
                return landmarks_sequence


        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None
        finally:
            # Ensure resources are released
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()

    def visualize_landmarks(self, frame, landmarks, view_angle="front", frame_number=None, output_path=None):
        """
        Visualize landmarks on the input frame with appropriate connections based on view angle

        Args:
            frame: Original video frame
            landmarks: Landmarks to visualize (numpy array)
            view_angle: Camera view angle ("front", "sag_left", or "sag_right")
            output_path: Optional path to save visualized frame

        Returns:
            visualization: Frame with landmarks visualization
        """
        try:
            # Create a copy of the frame to avoid modifying the original
            visualization = frame.copy()

            # If no landmarks detected, return the original frame
            if landmarks is None:
                return visualization

            h, w, _ = frame.shape

            # Log the view angle being used for debugging
            if frame_number % 100 == 0:
              logging.info(f"Visualizing landmarks with view angle: {view_angle}")

            # Define connections based on view angle
            if view_angle == "front":
                # For frontal view, show both sides equally
                body_connections = [
                    # Upper body
                    (11, 12),  # Left shoulder to right shoulder
                    (11, 13),  # Left shoulder to left elbow
                    (13, 15),  # Left elbow to left wrist
                    (12, 14),  # Right shoulder to right elbow
                    (14, 16),  # Right elbow to right wrist
                    (11, 23),  # Left shoulder to left hip
                    (12, 24),  # Right shoulder to right hip
                    (23, 24),  # Left hip to right hip

                    # Lower body
                    (23, 25),  # Left hip to left knee
                    (25, 27),  # Left knee to left ankle
                    (24, 26),  # Right hip to right knee
                    (26, 28),  # Right knee to right ankle
                    (27, 29),  # Left ankle to left heel
                    (29, 31),  # Left heel to left foot index
                    (27, 31),  # Left ankle to left foot index
                    (28, 30),  # Right ankle to right heel
                    (30, 32),  # Right heel to right foot index
                    (28, 32)  # Right ankle to right foot index
                ]

            elif view_angle == "sag_left":
                # For left sagittal view, emphasize left side connections
                body_connections = [
                    # Torso
                    (11, 23),  # Left shoulder to left hip

                    # Left arm
                    (11, 13),  # Left shoulder to left elbow
                    (13, 15),  # Left elbow to left wrist

                    # Left leg - IMPORTANT FOR SAGITTAL VIEW
                    (23, 25),  # Left hip to left knee
                    (25, 27),  # Left knee to left ankle
                    (27, 29),  # Left ankle to left heel
                    (29, 31),  # Left heel to left foot index
                    (27, 31),  # Left ankle to left foot index

                    # Additional postural connections
                    (0, 11),  # Nose to left shoulder (for head position)
                    (11, 12)  # Left to right shoulder (for orientation)
                ]

            elif view_angle == "sag_right":
                # For right sagittal view, emphasize right side connections
                body_connections = [
                    # Torso
                    (12, 24),  # Right shoulder to right hip

                    # Right arm
                    (12, 14),  # Right shoulder to right elbow
                    (14, 16),  # Right elbow to right wrist

                    # Right leg - IMPORTANT FOR SAGITTAL VIEW
                    (24, 26),  # Right hip to right knee
                    (26, 28),  # Right knee to right ankle
                    (28, 30),  # Right ankle to right heel
                    (30, 32),  # Right heel to right foot index
                    (28, 32),  # Right ankle to right foot index

                    # Additional postural connections
                    (0, 12),  # Nose to right shoulder (for head position)
                    (11, 12)  # Left to right shoulder (for orientation)
                ]
            else:
                # Default to front view if angle not recognized
                logging.warning(f"Unrecognized view angle: {view_angle}. Using front view as default.")
                body_connections = [
                    # Standard connections for frontal view
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24),
                    (23, 25), (25, 27), (24, 26), (26, 28),
                    (27, 29), (29, 31), (27, 31),
                    (28, 30), (30, 32), (28, 32)
                ]

            # Draw landmarks as circles
            landmark_size = 3  # Default circle size
            for i, landmark in enumerate(landmarks):
                # Only draw if visibility is above threshold
                if landmark[3] > 0.001:  # Using visibility threshold
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    # Draw circle for each landmark
                    cv2.circle(visualization, (x, y), landmark_size, (0, 255, 0), -1)
                    # Add landmark index for debugging/development
                    cv2.putText(visualization, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)

            # Draw body connections
            for connection in body_connections:
                start_idx, end_idx = connection
                if (start_idx < landmarks.shape[0] and end_idx < landmarks.shape[0] and
                        landmarks[start_idx][3] > 0.001 and landmarks[end_idx][3] > 0.001):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))

                    # Different colors for different body parts
                    if start_idx in [11, 12, 13, 14, 15, 16]:  # Upper body
                        color = (0, 0, 255)  # Red for upper body
                    else:  # Lower body
                        color = (255, 0, 0)  # Blue for lower body

                    cv2.line(visualization, start_point, end_point, color, 2)

            # Add view angle label to the frame
            cv2.putText(visualization, f"View: {view_angle}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Save visualization if output path is provided
            if output_path:
                ensure_directory_exists(os.path.dirname(output_path))
                cv2.imwrite(output_path, visualization)

            return visualization

        except Exception as e:
            logging.error(f"Error visualizing landmarks: {str(e)}")
            return frame  # Return original frame if visualization fails