# tests/test_mediapipe_estimator.py
import unittest
import os
import sys
import numpy as np
import cv2
from unittest import mock
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pose_estimators.mediapipe_estimator import MediaPipePoseEstimator
from src.core.config_manager import ConfigLoader


class TestMediaPipePoseEstimator(unittest.TestCase):
    """Test cases for MediaPipe pose estimator implementation"""

    def setUp(self):
        """Set up test environment before each test method"""
        # Load test configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config('default')

        # Create sample test frame
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure to have something to detect
        # (This might not trigger detection in real implementation)
        cv2.line(self.test_frame, (320, 100), (320, 300), (255, 255, 255), 5)  # Body
        cv2.line(self.test_frame, (320, 150), (250, 200), (255, 255, 255), 5)  # Left arm
        cv2.line(self.test_frame, (320, 150), (390, 200), (255, 255, 255), 5)  # Right arm
        cv2.line(self.test_frame, (320, 300), (250, 400), (255, 255, 255), 5)  # Left leg
        cv2.line(self.test_frame, (320, 300), (390, 400), (255, 255, 255), 5)  # Right leg
        cv2.circle(self.test_frame, (320, 100), 30, (255, 255, 255), -1)  # Head

    def test_initialization(self):
        """Test if the MediaPipe estimator initializes correctly with config"""
        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])

        # Verify MediaPipe was initialized
        self.assertIsNotNone(estimator.pose)

        # Verify configuration was applied
        mp_config = self.config['pose_estimation']['mediapipe']
        self.assertEqual(estimator.model_complexity, mp_config['model_complexity'])
        self.assertEqual(estimator.min_detection_confidence, mp_config['min_detection_confidence'])
        self.assertEqual(estimator.min_tracking_confidence, mp_config['min_tracking_confidence'])
        self.assertEqual(estimator.static_image_mode, mp_config['static_image_mode'])

    @mock.patch('mediapipe.solutions.pose.Pose')
    def test_extract_landmarks_with_detection(self, mock_pose):
        """Test landmark extraction when pose is detected"""
        # Create mock pose results with detected landmarks
        mock_landmark = mock.MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 0.9

        mock_results = mock.MagicMock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33  # 33 identical landmarks

        # Configure the mock to return our fake results
        mock_pose_instance = mock_pose.return_value
        mock_pose_instance.process.return_value = mock_results

        # Create estimator and extract landmarks
        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])
        landmarks = estimator.extract_landmarks(self.test_frame)

        # Verify correct dimensions and values
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.shape, (33, 4))  # 33 landmarks with x,y,z,visibility
        self.assertEqual(landmarks[0, 0], 0.5)  # x
        self.assertEqual(landmarks[0, 1], 0.5)  # y
        self.assertEqual(landmarks[0, 2], 0.0)  # z
        self.assertAlmostEqual(landmarks[0, 3], 0.9, places=5)  # visibility

        # Verify the mock was called with RGB image
        mock_pose_instance.process.assert_called_once()

    @mock.patch('mediapipe.solutions.pose.Pose')
    def test_extract_landmarks_no_detection(self, mock_pose):
        """Test landmark extraction when no pose is detected"""
        # Create mock pose results with no detected landmarks
        mock_results = mock.MagicMock()
        mock_results.pose_landmarks = None

        # Configure the mock to return our fake results
        mock_pose_instance = mock_pose.return_value
        mock_pose_instance.process.return_value = mock_results

        # Create estimator and extract landmarks
        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])
        landmarks = estimator.extract_landmarks(self.test_frame)

        # Verify no landmarks were returned
        self.assertIsNone(landmarks)


    @mock.patch('cv2.VideoCapture')
    @mock.patch('cv2.VideoWriter')
    @mock.patch('os.path.exists')
    def test_process_video(self, mock_exists, mock_writer, mock_capture):
        """Test video processing end-to-end"""
        # Mock os.path.exists to return True for our test video
        mock_exists.return_value = True

        # Mock the video capture
        mock_cap = mock_capture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 10
        }.get(prop, 0)

        # Mock read to return 5 frames then finish
        mock_cap.read.side_effect = [
                                        (True, self.test_frame)
                                        for _ in range(5)
                                    ] + [(False, None)]

        # Create a MediaPipe estimator with mocked extract_landmarks
        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])
        estimator.extract_landmarks = mock.MagicMock(
            return_value=np.zeros((33, 4), dtype=np.float32)
        )

        # Also mock visualize_landmarks to avoid dependencies
        estimator.visualize_landmarks = mock.MagicMock(
            return_value=self.test_frame
        )

        # Process video
        test_video_path = "dummy_video.mp4"
        test_output_path = "test_output"
        landmarks_sequence = estimator.process_video(
            test_video_path,
            test_output_path,
            video_segment={"start_frame": 0, "end_frame": 5}
        )

        # Verify correct methods were called
        mock_capture.assert_called_once_with(test_video_path)
        self.assertEqual(mock_cap.read.call_count, 6)  # 5 frames + 1 EOF check

        # Verify landmarks sequence shape
        self.assertIsNotNone(landmarks_sequence)
        self.assertEqual(landmarks_sequence.shape, (5, 33, 4))

        # Verify extract_landmarks was called for each frame
        self.assertEqual(estimator.extract_landmarks.call_count, 5)

    def test_visualize_landmarks(self):
        """Test landmark visualization"""
        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])

        # Create dummy landmarks that are valid (33 landmarks with x, y, z, visibility)
        landmarks = np.zeros((33, 4), dtype=np.float32)
        for i in range(33):
            landmarks[i, 0] = 0.5  # x
            landmarks[i, 1] = 0.5  # y
            landmarks[i, 2] = 0.0  # z
            landmarks[i, 3] = 0.9  # visibility

        # Call visualize_landmarks
        vis_frame = estimator.visualize_landmarks(self.test_frame, landmarks)

        # Verify that visualization is returned and has same dimensions as input
        self.assertIsNotNone(vis_frame)
        self.assertEqual(vis_frame.shape, self.test_frame.shape)

        # Verify landmarks were drawn (changes were made to the frame)
        self.assertFalse(np.array_equal(vis_frame, self.test_frame))

    @mock.patch('os.path.exists')
    def test_process_video_missing_file(self, mock_exists):
        """Test behavior when video file doesn't exist"""

        # Configure mock to return False for our test file but True for anything else
        def mock_exists_side_effect(path):
            if path == "nonexistent_video.mp4":
                return False
            return True  # Return True for MediaPipe model files

        mock_exists.side_effect = mock_exists_side_effect

        estimator = MediaPipePoseEstimator(self.config['pose_estimation'])
        result = estimator.process_video("nonexistent_video.mp4")

        self.assertIsNone(result)
        # Verify our specific path was checked
        mock_exists.assert_any_call("nonexistent_video.mp4")


if __name__ == '__main__':
    unittest.main()