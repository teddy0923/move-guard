# scripts/generate_visualizations.py
# !/usr/bin/env python3
"""
Script to generate visualizations of movement analysis with ankle angle overlays.
Simplified to focus on ankle dorsiflexion angle only.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_landmarks, ensure_directory_exists, load_metadata, load_video_metadata_file
from src.core.pipeline import Pipeline


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate movement analysis visualizations with ankle angle')

    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--landmarks', type=str, required=True,
                        help='Path to landmarks file (.npy)')
    parser.add_argument('--output', type=str,
                        help='Output directory for visualization videos')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata CSV file with optional start/end frames')
    parser.add_argument('--angle-data', type=str,
                        help='Path to pre-calculated angle data CSV file')
    parser.add_argument('--features', type=str,
                        help='Path to extracted features file to visualize metrics')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')
    parser.add_argument('--movement', type=str, default='squat',
                        help='Movement type (e.g., squat, ybt)')
    parser.add_argument('--fps', type=int,
                        help='Output video FPS (defaults to original video FPS)')
    parser.add_argument('--show-frames', action='store_true',
                        help='Show frame numbers on the visualization')
    parser.add_argument('--hide-landmarks', action='store_true',
                        help='Hide landmark dots and connections')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args(args)


def draw_landmarks(frame, landmarks, connections, color=(0, 255, 0), thickness=2, visibility_threshold=0.001):
    """Draw landmarks and connections on frame"""
    h, w = frame.shape[:2]

    # Draw landmarks
    for i, landmark in enumerate(landmarks):
        # Skip if less than 3 dimensions or low visibility
        if len(landmark) < 3 or (len(landmark) > 3 and landmark[3] < visibility_threshold):
            continue

        # Convert normalized coordinates to pixel values
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(frame, (x, y), 5, color, -1)

        # Optional: add landmark index for debugging
        # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection

        # Skip if indices are outside the range
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        # Skip if landmarks have less than 3 dimensions or low visibility
        if len(landmarks[start_idx]) < 3 or len(landmarks[end_idx]) < 3:
            continue

        if len(landmarks[start_idx]) > 3 and len(landmarks[end_idx]) > 3:
            if landmarks[start_idx][3] < visibility_threshold or landmarks[end_idx][3] < visibility_threshold:
                continue

        start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
        end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
        cv2.line(frame, start_point, end_point, color, thickness)

    return frame


def draw_ankle_angle(frame, landmarks, view_angle, angle_value=None, color=(0, 255, 255), thickness=2, debug=False):
    """Draw ankle angle visualization with pre-calculated value"""
    h, w = frame.shape[:2]

    if debug:
        print(f"Drawing ankle angle for view: {view_angle}")
        print(f"Using pre-calculated angle value: {angle_value}")

    # Determine which side to use based on view_angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        heel_idx = 29  # left_heel
        foot_idx = 31  # left_foot_index
        ankle_idx = 27  # left_ankle
        knee_idx = 25  # left_knee
        if debug:
            print(f"Using LEFT side landmarks: heel={heel_idx}, foot={foot_idx}, ankle={ankle_idx}, knee={knee_idx}")
    else:  # For sagittal right or default
        heel_idx = 30  # right_heel
        foot_idx = 32  # right_foot_index
        ankle_idx = 28  # right_ankle
        knee_idx = 26  # right_knee
        if debug:
            print(f"Using RIGHT side landmarks: heel={heel_idx}, foot={foot_idx}, ankle={ankle_idx}, knee={knee_idx}")

    # Check visibility of all required landmarks
    visibility_threshold = 0.0001  # Even lower threshold for better detection

    if (heel_idx >= len(landmarks) or foot_idx >= len(landmarks) or
            ankle_idx >= len(landmarks) or knee_idx >= len(landmarks)):
        if debug:
            print(f"Landmark indices out of range. Landmarks shape: {landmarks.shape}")
        return frame

    # Print visibility values for debugging
    if debug:
        visibility_values = []
        if len(landmarks[heel_idx]) > 3:
            visibility_values.append(("heel", landmarks[heel_idx][3]))
        if len(landmarks[foot_idx]) > 3:
            visibility_values.append(("foot", landmarks[foot_idx][3]))
        if len(landmarks[ankle_idx]) > 3:
            visibility_values.append(("ankle", landmarks[ankle_idx][3]))
        if len(landmarks[knee_idx]) > 3:
            visibility_values.append(("knee", landmarks[knee_idx][3]))
        print(f"Landmark visibility values: {visibility_values}")

    # Skip visibility checks during debugging to see what's happening
    skip_visibility_check = debug

    if not skip_visibility_check:
        if len(landmarks[heel_idx]) > 3 and landmarks[heel_idx][3] < visibility_threshold:
            if debug:
                print(f"Heel visibility too low: {landmarks[heel_idx][3]}")
            return frame
        if len(landmarks[foot_idx]) > 3 and landmarks[foot_idx][3] < visibility_threshold:
            if debug:
                print(f"Foot visibility too low: {landmarks[foot_idx][3]}")
            return frame
        if len(landmarks[ankle_idx]) > 3 and landmarks[ankle_idx][3] < visibility_threshold:
            if debug:
                print(f"Ankle visibility too low: {landmarks[ankle_idx][3]}")
            return frame
        if len(landmarks[knee_idx]) > 3 and landmarks[knee_idx][3] < visibility_threshold:
            if debug:
                print(f"Knee visibility too low: {landmarks[knee_idx][3]}")
            return frame

    # Get coordinates for the ankle angle points
    heel_pos = (int(landmarks[heel_idx][0] * w), int(landmarks[heel_idx][1] * h))
    foot_pos = (int(landmarks[foot_idx][0] * w), int(landmarks[foot_idx][1] * h))
    ankle_pos = (int(landmarks[ankle_idx][0] * w), int(landmarks[ankle_idx][1] * h))
    knee_pos = (int(landmarks[knee_idx][0] * w), int(landmarks[knee_idx][1] * h))

    if debug:
        print(f"Positions: heel={heel_pos}, foot={foot_pos}, ankle={ankle_pos}, knee={knee_pos}")

    # Draw the segments that form the ankle angle with thicker lines
    cv2.line(frame, heel_pos, ankle_pos, color, thickness)
    cv2.line(frame, foot_pos, ankle_pos, color, thickness)
    cv2.line(frame, knee_pos, ankle_pos, (255, 0, 255), thickness)  # Different color

    # Draw the ankle angle value directly at the ankle joint
    if angle_value is not None:
        # Calculate text position - make sure it doesn't overlap with the joint
        text_x = ankle_pos[0] + 10
        text_y = ankle_pos[1] - 20  # Position above the ankle

        # Draw text with black background for better visibility
        text = f"{angle_value:.1f} deg"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(frame,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)

        if debug:
            print(f"Drew angle text: {text} at position {(text_x, text_y)}")
    else:
        # Calculate angle on the fly
        try:
            # Ankle angle between heel-foot and ankle-knee segments
            heel_foot_vector = np.array([foot_pos[0] - heel_pos[0], foot_pos[1] - heel_pos[1]])
            ankle_knee_vector = np.array([knee_pos[0] - ankle_pos[0], knee_pos[1] - ankle_pos[1]])

            heel_foot_mag = np.linalg.norm(heel_foot_vector)
            ankle_knee_mag = np.linalg.norm(ankle_knee_vector)

            if heel_foot_mag > 0 and ankle_knee_mag > 0:
                dot_product = np.dot(heel_foot_vector, ankle_knee_vector)
                cos_angle = max(min(dot_product / (heel_foot_mag * ankle_knee_mag), 1.0), -1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                if debug:
                    print(f"Calculated angle: {angle_deg:.1f}°")

                text = f"{angle_deg:.1f}°"
                text_x = ankle_pos[0] + 10
                text_y = ankle_pos[1] - 20

                # Draw text with black background for better visibility
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, thickness)
                cv2.rectangle(frame,
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
        except Exception as e:
            if debug:
                print(f"Error calculating ankle angle: {str(e)}")
            logging.error(f"Error calculating ankle angle: {str(e)}")

    return frame


def draw_knee_angle(frame, landmarks, view_angle, angle_value=None, color=(255, 165, 0), thickness=2, debug=False):
    """Draw knee angle visualization with pre-calculated value"""
    h, w = frame.shape[:2]

    if debug:
        print(f"Drawing knee angle for view: {view_angle}")
        print(f"Using pre-calculated angle value: {angle_value}")

    # Determine which side to use based on view_angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        hip_idx = 23  # left_hip
        knee_idx = 25  # left_knee
        ankle_idx = 27  # left_ankle
    else:  # For sagittal right or default
        hip_idx = 24  # right_hip
        knee_idx = 26  # right_knee
        ankle_idx = 28  # right_ankle

    if debug:
        print(f"Using landmarks: hip={hip_idx}, knee={knee_idx}, ankle={ankle_idx}")

    # Check if indices are valid
    if (hip_idx >= len(landmarks) or knee_idx >= len(landmarks) or ankle_idx >= len(landmarks)):
        if debug:
            print(f"Landmark indices out of range. Landmarks shape: {landmarks.shape}")
        return frame

    # Skip visibility checks during debugging
    skip_visibility_check = debug
    visibility_threshold = 0.0001

    if not skip_visibility_check:
        for idx in [hip_idx, knee_idx, ankle_idx]:
            if len(landmarks[idx]) > 3 and landmarks[idx][3] < visibility_threshold:
                if debug:
                    print(f"Landmark {idx} visibility too low: {landmarks[idx][3]}")
                return frame

    # Get coordinates
    hip_pos = (int(landmarks[hip_idx][0] * w), int(landmarks[hip_idx][1] * h))
    knee_pos = (int(landmarks[knee_idx][0] * w), int(landmarks[knee_idx][1] * h))
    ankle_pos = (int(landmarks[ankle_idx][0] * w), int(landmarks[ankle_idx][1] * h))

    if debug:
        print(f"Positions: hip={hip_pos}, knee={knee_pos}, ankle={ankle_pos}")

    # Draw the segments that form the knee angle
    cv2.line(frame, hip_pos, knee_pos, color, thickness)
    cv2.line(frame, ankle_pos, knee_pos, color, thickness)

    # Draw the knee angle value
    if angle_value is not None:
        text_x = knee_pos[0] + 10
        text_y = knee_pos[1] - 20

        text = f"{angle_value:.1f} deg"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(frame,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
    else:
        # Calculate angle on the fly
        try:
            hip_knee_vector = np.array([hip_pos[0] - knee_pos[0], hip_pos[1] - knee_pos[1]])
            ankle_knee_vector = np.array([ankle_pos[0] - knee_pos[0], ankle_pos[1] - knee_pos[1]])

            hip_knee_mag = np.linalg.norm(hip_knee_vector)
            ankle_knee_mag = np.linalg.norm(ankle_knee_vector)

            if hip_knee_mag > 0 and ankle_knee_mag > 0:
                dot_product = np.dot(hip_knee_vector, ankle_knee_vector)
                cos_angle = max(min(dot_product / (hip_knee_mag * ankle_knee_mag), 1.0), -1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                if debug:
                    print(f"Calculated knee angle: {angle_deg:.1f}°")

                text = f"{angle_deg:.1f}°"
                text_x = knee_pos[0] + 10
                text_y = knee_pos[1] - 20

                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, thickness)
                cv2.rectangle(frame,
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
        except Exception as e:
            if debug:
                print(f"Error calculating knee angle: {str(e)}")
            logging.error(f"Error calculating knee angle: {str(e)}")

    return frame


def draw_hip_flexion(frame, landmarks, view_angle, angle_value=None, color=(0, 120, 255), thickness=2, debug=False):
    """Draw hip flexion angle visualization with pre-calculated value"""
    h, w = frame.shape[:2]

    if debug:
        print(f"Drawing hip flexion for view: {view_angle}")
        print(f"Using pre-calculated angle value: {angle_value}")

    # Determine which side to use based on view_angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        shoulder_idx = 11  # left_shoulder
        hip_idx = 23  # left_hip
        knee_idx = 25  # left_knee
    else:  # For sagittal right or default
        shoulder_idx = 12  # right_shoulder
        hip_idx = 24  # right_hip
        knee_idx = 26  # right_knee

    if debug:
        print(f"Using landmarks: shoulder={shoulder_idx}, hip={hip_idx}, knee={knee_idx}")

    # Check if indices are valid
    if (shoulder_idx >= len(landmarks) or hip_idx >= len(landmarks) or knee_idx >= len(landmarks)):
        if debug:
            print(f"Landmark indices out of range. Landmarks shape: {landmarks.shape}")
        return frame

    # Skip visibility checks during debugging
    skip_visibility_check = debug
    visibility_threshold = 0.0001

    if not skip_visibility_check:
        for idx in [shoulder_idx, hip_idx, knee_idx]:
            if len(landmarks[idx]) > 3 and landmarks[idx][3] < visibility_threshold:
                if debug:
                    print(f"Landmark {idx} visibility too low: {landmarks[idx][3]}")
                return frame

    # Get coordinates
    shoulder_pos = (int(landmarks[shoulder_idx][0] * w), int(landmarks[shoulder_idx][1] * h))
    hip_pos = (int(landmarks[hip_idx][0] * w), int(landmarks[hip_idx][1] * h))
    knee_pos = (int(landmarks[knee_idx][0] * w), int(landmarks[knee_idx][1] * h))

    if debug:
        print(f"Positions: shoulder={shoulder_pos}, hip={hip_pos}, knee={knee_pos}")

    # Draw the segments that form the hip angle
    cv2.line(frame, shoulder_pos, hip_pos, color, thickness)
    cv2.line(frame, knee_pos, hip_pos, color, thickness)

    # Draw the hip angle value
    if angle_value is not None:
        text_x = hip_pos[0] + 10
        text_y = hip_pos[1] - 20

        text = f"{angle_value:.1f} deg"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(frame,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
    else:
        try:
            # Calculate vectors
            hip_shoulder_vector = np.array([shoulder_pos[0] - hip_pos[0], shoulder_pos[1] - hip_pos[1]])
            hip_knee_vector = np.array([knee_pos[0] - hip_pos[0], knee_pos[1] - hip_pos[1]])

            hip_shoulder_mag = np.linalg.norm(hip_shoulder_vector)
            hip_knee_mag = np.linalg.norm(hip_knee_vector)

            if hip_shoulder_mag > 0 and hip_knee_mag > 0:
                dot_product = np.dot(hip_shoulder_vector, hip_knee_vector)
                cos_angle = max(min(dot_product / (hip_shoulder_mag * hip_knee_mag), 1.0), -1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                if debug:
                    print(f"Calculated hip flexion angle: {angle_deg:.1f}°")

                text = f"{angle_deg:.1f}°"
                text_x = hip_pos[0] + 10
                text_y = hip_pos[1] - 20

                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, thickness)
                cv2.rectangle(frame,
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
        except Exception as e:
            if debug:
                print(f"Error calculating hip flexion angle: {str(e)}")
            logging.error(f"Error calculating hip flexion angle: {str(e)}")

    return frame


def draw_shoulder_flexion(frame, landmarks, view_angle, angle_value=None, color=(255, 0, 255), thickness=2, debug=False):
    """Draw shoulder flexion angle visualization with pre-calculated value"""
    h, w = frame.shape[:2]

    if debug:
        print(f"Drawing shoulder flexion for view: {view_angle}")
        print(f"Using pre-calculated angle value: {angle_value}")

    # Determine which side to use based on view_angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        hip_idx = 23  # left_hip
        shoulder_idx = 11  # left_shoulder
        elbow_idx = 13  # left_elbow
    else:  # For sagittal right or default
        hip_idx = 24  # right_hip
        shoulder_idx = 12  # right_shoulder
        elbow_idx = 14  # right_elbow

    if debug:
        print(f"Using landmarks: hip={hip_idx}, shoulder={shoulder_idx}, elbow={elbow_idx}")

    # Check if indices are valid
    if (hip_idx >= len(landmarks) or shoulder_idx >= len(landmarks) or elbow_idx >= len(landmarks)):
        if debug:
            print(f"Landmark indices out of range. Landmarks shape: {landmarks.shape}")
        return frame

    # Skip visibility checks during debugging
    skip_visibility_check = debug
    visibility_threshold = 0.0001

    if not skip_visibility_check:
        for idx in [hip_idx, shoulder_idx, elbow_idx]:
            if len(landmarks[idx]) > 3 and landmarks[idx][3] < visibility_threshold:
                if debug:
                    print(f"Landmark {idx} visibility too low: {landmarks[idx][3]}")
                return frame

    # Get coordinates
    hip_pos = (int(landmarks[hip_idx][0] * w), int(landmarks[hip_idx][1] * h))
    shoulder_pos = (int(landmarks[shoulder_idx][0] * w), int(landmarks[shoulder_idx][1] * h))
    elbow_pos = (int(landmarks[elbow_idx][0] * w), int(landmarks[elbow_idx][1] * h))

    if debug:
        print(f"Positions: hip={hip_pos}, shoulder={shoulder_pos}, elbow={elbow_pos}")

    # Draw the segments that form the shoulder angle
    cv2.line(frame, hip_pos, shoulder_pos, color, thickness)
    cv2.line(frame, elbow_pos, shoulder_pos, color, thickness)

    # Draw the shoulder angle value
    if angle_value is not None:
        text_pos = (shoulder_pos[0] + 10, shoulder_pos[1] - 10)  # Position text near shoulder
        text = f"{angle_value:.1f}°"
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    else:
        try:
            # Calculate vectors
            shoulder_hip_vector = np.array([hip_pos[0] - shoulder_pos[0], hip_pos[1] - shoulder_pos[1]])
            shoulder_elbow_vector = np.array([elbow_pos[0] - shoulder_pos[0], elbow_pos[1] - shoulder_pos[1]])

            shoulder_hip_mag = np.linalg.norm(shoulder_hip_vector)
            shoulder_elbow_mag = np.linalg.norm(shoulder_elbow_vector)

            if shoulder_hip_mag > 0 and shoulder_elbow_mag > 0:
                dot_product = np.dot(shoulder_hip_vector, shoulder_elbow_vector)
                cos_angle = max(min(dot_product / (shoulder_hip_mag * shoulder_elbow_mag), 1.0), -1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                if debug:
                    print(f"Calculated shoulder flexion angle: {angle_deg:.1f}°")

                text = f"{angle_deg:.1f}°"
                text_pos = (shoulder_pos[0] + 10, shoulder_pos[1] - 10)  # Position text near shoulder
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        except Exception as e:
            if debug:
                print(f"Error calculating shoulder flexion angle: {str(e)}")
            logging.error(f"Error calculating shoulder flexion angle: {str(e)}")

    return frame


def draw_femoral_angle(frame, landmarks, view_angle, angle_value=None, color=(100, 255, 100), thickness=2, debug=False):
    """Draw femoral angle visualization (angle between femur and horizontal)"""
    h, w = frame.shape[:2]

    if debug:
        print(f"Drawing femoral angle for view: {view_angle}")
        print(f"Using pre-calculated angle value: {angle_value}")

    # Determine which side to use based on view_angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        knee_idx = 25  # left_knee
        hip_idx = 23  # left_hip
    else:  # For sagittal right or default
        knee_idx = 26  # right_knee
        hip_idx = 24  # right_hip

    if debug:
        print(f"Using landmarks: knee={knee_idx}, hip={hip_idx}")

    # Check if indices are valid
    if (knee_idx >= len(landmarks) or hip_idx >= len(landmarks)):
        if debug:
            print(f"Landmark indices out of range. Landmarks shape: {landmarks.shape}")
        return frame

    # Skip visibility checks during debugging
    skip_visibility_check = debug
    visibility_threshold = 0.0001

    if not skip_visibility_check:
        for idx in [knee_idx, hip_idx]:
            if len(landmarks[idx]) > 3 and landmarks[idx][3] < visibility_threshold:
                if debug:
                    print(f"Landmark {idx} visibility too low: {landmarks[idx][3]}")
                return frame

    # Get coordinates
    knee_pos = (int(landmarks[knee_idx][0] * w), int(landmarks[knee_idx][1] * h))
    hip_pos = (int(landmarks[hip_idx][0] * w), int(landmarks[hip_idx][1] * h))

    if debug:
        print(f"Positions: knee={knee_pos}, hip={hip_pos}")

    # Draw horizontal reference line
    horizontal_start = (knee_pos[0] - 50, knee_pos[1])  # 50 pixels to the left of knee
    horizontal_end = (knee_pos[0] + 50, knee_pos[1])    # 50 pixels to the right of knee
    cv2.line(frame, horizontal_start, horizontal_end, color, thickness, cv2.LINE_AA)

    # Draw femur line (knee to hip)
    cv2.line(frame, knee_pos, hip_pos, color, thickness)

    # Draw the angle value
    if angle_value is not None:
        # Position text near the knee joint
        text_pos = (knee_pos[0] + 60, knee_pos[1])  # Offset to the right of the knee
        text = f"{angle_value:.1f}°"
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    else:
        try:
            # Calculate angle on the fly
            femur_vector = np.array([hip_pos[0] - knee_pos[0], hip_pos[1] - knee_pos[1]])
            horizontal_vector = np.array([1, 0])  # Horizontal vector

            femur_mag = np.linalg.norm(femur_vector)
            horizontal_mag = np.linalg.norm(horizontal_vector)

            if femur_mag > 0 and horizontal_mag > 0:
                dot_product = np.dot(femur_vector, horizontal_vector)
                cos_angle = max(min(dot_product / (femur_mag * horizontal_mag), 1.0), -1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                if debug:
                    print(f"Calculated femoral angle: {angle_deg:.1f}°")

                text = f"{angle_deg:.1f}°"
                text_pos = (knee_pos[0] + 60, knee_pos[1])  # Offset to the right of the knee
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        except Exception as e:
            if debug:
                print(f"Error calculating femoral angle: {str(e)}")
            logging.error(f"Error calculating femoral angle: {str(e)}")

    return frame


def get_landmark_indices():
    """Get indices for MediaPipe landmarks"""
    return {
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


def main():
    # Parse arguments
    args = parse_args()

    # Set up logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Load configuration
    config_loader = ConfigLoader()
    base_config = config_loader.load_config(args.config)

    if not base_config:
        logging.error(f"Failed to load base configuration: {args.config}")
        sys.exit(1)

    # Load movement-specific configuration
    movement_config = config_loader.load_config(args.movement)
    if movement_config:
        config = config_loader.merge_configs(args.config, args.movement)
    else:
        config = base_config

    # Set default output directory if not specified
    if not args.output:
        args.output = base_config['paths']['data']['processed_videos']['annotated']

    # Ensure output directory exists
    ensure_directory_exists(args.output)

    # Load landmarks
    landmarks_data = load_landmarks(args.landmarks)
    if landmarks_data is None:
        logging.error(f"Failed to load landmarks from {args.landmarks}")
        sys.exit(1)

    logging.info(f"Loaded landmarks with shape: {landmarks_data.shape}")

    # Load video
    video_path = args.video
    video_id = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        sys.exit(1)

    # Get video properties
    fps = args.fps if args.fps else int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine start and end frames
    start_frame = 0
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Get landmark indices
    landmark_indices = get_landmark_indices()

    # Define view angle (default to "Sagittal Left" for safety)
    view_angle = "Sagittal Left"  # Change default to ensure we get ankle angle display

    # Load metadata if provided
    metadata_dict = None
    if args.metadata:
        metadata_dict = load_video_metadata_file(args.metadata, config)

        if metadata_dict:
            # Try to find metadata for this video (with and without extension)
            metadata_entry = None
            if video_id in metadata_dict:
                metadata_entry = metadata_dict[video_id]
                logging.info(f"Found metadata for {video_id}")
            elif f"{video_id}.mp4" in metadata_dict:
                metadata_entry = metadata_dict[f"{video_id}.mp4"]
                logging.info(f"Found metadata for {video_id}.mp4")
            else:
                logging.warning(f"No metadata found for video ID: {video_id}")
                if args.debug:
                    print(f"Available video IDs in metadata: {list(metadata_dict.keys())}")

            if metadata_entry:
                # Extract view angle and convert to more readable format
                if 'View_Angle' in metadata_entry:
                    raw_view = metadata_entry['View_Angle']
                    logging.info(f"Raw view angle from metadata: '{raw_view}'")

                    if args.debug:
                        print(f"Raw view angle from metadata: '{raw_view}'")

                    if raw_view.lower() in ["sag_left", "sagittal_left", "sagittal left"]:
                        view_angle = "Sagittal Left"
                    elif raw_view.lower() in ["sag_right", "sagittal_right", "sagittal right"]:
                        view_angle = "Sagittal Right"
                    elif raw_view.lower() in ["front", "frontal"]:
                        view_angle = "Front"
                    else:
                        view_angle = raw_view  # Keep as is if not recognized

                    logging.info(f"Determined view angle: {view_angle}")

                # Extract start/end frames for repetitions
                if 'repetitions' in metadata_entry and metadata_entry['repetitions']:
                    # Use first repetition for visualization boundaries
                    first_rep = metadata_entry['repetitions'][0]
                    if 'start_frame' in first_rep and 'end_frame' in first_rep:
                        start_frame = first_rep['start_frame']
                        end_frame = first_rep['end_frame']
                        logging.info(f"Using frames {start_frame} to {end_frame} from metadata repetition")

    # Override view angle if needed for debugging
    if args.debug:
        # Force view angle to sagittal for testing
        if view_angle.lower() == "front":
            logging.warning("Overriding 'front' view to 'Sagittal Left' for testing")
            view_angle = "Sagittal Left"

    # Load pre-calculated angle data if provided
    angle_data = None
    if args.angle_data:
        try:
            angles_df = pd.read_csv(args.angle_data)
            logging.info(f"Loaded angle data CSV with shape: {angles_df.shape}")

            if args.debug:
                print(f"Angle data columns: {angles_df.columns.tolist()}")
                if len(angles_df) > 0:
                    print(f"First row: {angles_df.iloc[0].to_dict()}")

            if 'frame' in angles_df.columns:
                angle_data = {}
                for _, row in angles_df.iterrows():
                    frame_idx = int(row['frame'])
                    frame_angles = {}
                    
                    # Map CSV columns to our internal names
                    if 'ankle_angle' in angles_df.columns:
                        frame_angles['ankle'] = float(row['ankle_angle'])
                    if 'knee_angle' in angles_df.columns:
                        frame_angles['knee'] = float(row['knee_angle'])
                    if 'hip_flexion' in angles_df.columns:
                        frame_angles['hip_flexion'] = float(row['hip_flexion'])
                    if 'shoulder_flexion' in angles_df.columns:
                        frame_angles['shoulder_flexion'] = float(row['shoulder_flexion'])
                    if 'femoral_angle' in angles_df.columns:
                        frame_angles['femoral_angle'] = float(row['femoral_angle'])
                    
                    angle_data[frame_idx] = frame_angles
                
                logging.info(f"Loaded angles for {len(angle_data)} frames")
                if args.debug:
                    first_items = list(angle_data.items())[:5]
                    print(f"First 5 angle values: {first_items}")
                    print(f"Available angle columns: {[col for col in angles_df.columns]}")
            else:
                logging.warning("Angle data CSV missing required 'frame' column")
        except Exception as e:
            logging.error(f"Error loading angle data: {str(e)}")

    # Load feature data if provided
    feature_data = None
    if args.features:
        try:
            features_df = pd.read_csv(args.features)
            if not features_df.empty:
                feature_data = features_df.iloc[0].to_dict()  # Use first row for now
                logging.info(f"Loaded feature data: {feature_data}")
        except Exception as e:
            logging.error(f"Error loading feature data: {str(e)}")

    # Validate landmark-video correspondence
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_landmarks = len(landmarks_data)

    if total_landmarks < (end_frame - start_frame + 1):
        logging.warning(f"Video has {end_frame - start_frame + 1} frames in the specified range, "
                        f"but landmarks data only has {total_landmarks} frames")
        end_frame = start_frame + total_landmarks - 1
        logging.warning(f"Adjusting end_frame to {end_frame}")

    # Setup video writer
    output_path = os.path.join(args.output, f"{video_id}_ankle_angle.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define body connections for visualization based on view angle
    if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left"]:
        # Left side connections only - for sagittal left view
        body_connections = [
            # Torso - only left side
            (11, 23),  # Left shoulder to left hip
            # Left arm
            (11, 13), (13, 15),  # Left shoulder -> elbow -> wrist
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # Left hip -> knee -> ankle -> heel/foot
        ]
    elif view_angle.lower() in ["sag_right", "sagittal right", "sagittal_right"]:
        # Right side connections only - for sagittal right view
        body_connections = [
            # Torso - only right side
            (12, 24),  # Right shoulder to right hip
            # Right arm
            (12, 14), (14, 16),  # Right shoulder -> elbow -> wrist
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # Right hip -> knee -> ankle -> heel/foot
        ]
    else:
        # Frontal view - show both sides
        body_connections = [
            # Upper body
            (11, 12),  # Left shoulder to right shoulder
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hip line
            # Lower body
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (27, 29), (29, 31), (27, 31),  # Left foot
            (28, 30), (30, 32), (28, 32)  # Right foot
        ]

    # Process video
    frame_idx = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames before start_frame
        if frame_idx < start_frame:
            frame_idx += 1
            continue

        # Stop after end_frame
        if frame_idx > end_frame:
            break

        # Get landmarks for current frame
        landmarks_idx = frame_idx - start_frame
        if landmarks_idx < len(landmarks_data):
            frame_landmarks = landmarks_data[landmarks_idx]

            # Draw landmarks and connections if not hidden
            if not args.hide_landmarks:
                frame = draw_landmarks(frame, frame_landmarks, body_connections)

            # Draw angles for sagittal views
            if view_angle.lower() in ["sag_left", "sagittal left", "sagittal_left",
                                    "sag_right", "sagittal right", "sagittal_right"]:
                # Get pre-calculated angle values if available
                ankle_value = None
                knee_value = None
                hip_value = None
                shoulder_value = None
                femoral_value = None

                if angle_data and frame_idx in angle_data:
                    frame_angles = angle_data[frame_idx]
                    ankle_value = frame_angles.get('ankle')
                    knee_value = frame_angles.get('knee')
                    hip_value = frame_angles.get('hip_flexion')
                    shoulder_value = frame_angles.get('shoulder_flexion')
                    femoral_value = frame_angles.get('femoral_angle')

                    if args.debug and frame_idx % 50 == 0:
                        print(f"Frame {frame_idx}:")
                        print(f"  View angle: {view_angle}")
                        print(f"  Frame angles: {frame_angles}")
                        print(f"  Values: ankle={ankle_value}, knee={knee_value}, hip={hip_value}, shoulder={shoulder_value}, femoral={femoral_value}")

                # Draw angles and their values
                frame = draw_ankle_angle(frame, frame_landmarks, view_angle, ankle_value, debug=args.debug)
                frame = draw_knee_angle(frame, frame_landmarks, view_angle, knee_value, debug=args.debug)
                frame = draw_hip_flexion(frame, frame_landmarks, view_angle, hip_value, debug=args.debug)
                frame = draw_shoulder_flexion(frame, frame_landmarks, view_angle, shoulder_value, debug=args.debug)
                frame = draw_femoral_angle(frame, frame_landmarks, view_angle, femoral_value, debug=args.debug)

            # Add angle labels in top-left corner with black background
            if angle_data and frame_idx in angle_data:
                frame_angles = angle_data[frame_idx]
                ankle_value = frame_angles.get('ankle')
                knee_value = frame_angles.get('knee')
                hip_value = frame_angles.get('hip_flexion')
                shoulder_value = frame_angles.get('shoulder_flexion')
                femoral_value = frame_angles.get('femoral_angle')

                y_offset = 40
                angle_info = [
                    ("Dorsiflexion", ankle_value, (255, 255, 255)),
                    ("Knee Flexion", knee_value, (255, 165, 0)),
                    ("Hip Flexion", hip_value, (0, 120, 255)),
                    ("Shoulder Flexion", shoulder_value, (255, 0, 255)),
                    ("Femoral-Horizontal", femoral_value, (100, 255, 100))
                ]

                for label, value, color in angle_info:
                    if value is not None:
                        text = f"{label}: {value:.1f}°"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        
                        # Draw black background
                        cv2.rectangle(frame,
                                    (10, y_offset - text_size[1] - 5),
                                    (10 + text_size[0] + 10, y_offset + 5),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(frame, text, (15, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                        y_offset += 30

            # Add frame number if enabled
            if args.show_frames:
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            # Add view angle indicator - make sure it's using the correct view
            cv2.putText(frame, f"View: {view_angle}", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)  # Changed color to cyan for better visibility

        # Write frame to output video
        out.write(frame)

        frame_idx += 1
        processed_frames += 1

        if frame_idx % 100 == 0:
            logging.info(f"Processed {processed_frames} of {end_frame - start_frame + 1} frames")

    # Release resources
    cap.release()
    out.release()

    logging.info(f"Visualization saved to {output_path}")
    logging.info(f"Total frames processed: {processed_frames}")


if __name__ == "__main__":
    main()