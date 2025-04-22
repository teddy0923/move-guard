#!/usr/bin/env python3
"""
Script to convert landmarks.npy to a readable text format with X, Y, Z coordinates and visibility scores
"""

import numpy as np
import argparse
from pathlib import Path


def convert_landmarks(input_path: str, output_path: str):
    # Load the 4D numpy array
    data = np.load(input_path)
    
    # Get dimensions
    frames, landmarks, coords = data.shape
    
    # Open file in write mode
    with open(output_path, 'w') as f:
        # Write header with X, Y, Z, visibility for each landmark
        header_parts = []
        for i in range(landmarks):
            header_parts.extend([f'landmark{i}_x', f'landmark{i}_y', f'landmark{i}_z', f'landmark{i}_visibility'])
        f.write(f"frame,{','.join(header_parts)}\n")
        
        # Process each frame
        for frame_idx in range(frames):
            # Create row with frame number first
            frame_data = data[frame_idx].flatten()
            row = f"{frame_idx}," + ",".join(f"{x:.6f}" for x in frame_data)
            f.write(row + "\n")
    
    print(f"Converted {input_path} to {output_path}")
    print(f"Data shape: {data.shape}")
    print(f"Each frame has {landmarks} landmarks with {coords} values (X, Y, Z, visibility)")


def main():
    parser = argparse.ArgumentParser(description='Convert landmarks.npy to text format')
    parser.add_argument('input', help='Path to input .npy file')
    parser.add_argument('--output', '-o', help='Path to output text file', default='landmarks.txt')
    
    args = parser.parse_args()
    
    convert_landmarks(args.input, args.output)


if __name__ == "__main__":
    main()