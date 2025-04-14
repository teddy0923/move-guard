# tests/test_real_data.py
import os
import sys
import pytest
from pathlib import Path


# Add the project root to the path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigLoader
from src.core.file_utils import load_video_metadata_file


class TestRealData:
    """Tests using actual project data files"""

    @pytest.fixture
    def config(self):
        """Load configuration for tests"""
        config_loader = ConfigLoader(config_dir=os.path.join(project_root, 'config'))
        return config_loader.load_config('default')


    def test_metadata_loading(self, config):
        """Test loading the actual metadata CSV file"""
        # Try multiple possible locations for the metadata file
        possible_paths = [
            # From config
            os.path.join(project_root, config['paths']['data']['metadata']['test_data']),
            # Directly in project root
            os.path.join(project_root, 'tests_data.csv'),
            # In data/metadata folder
            os.path.join(project_root, 'data', 'metadata', 'tests_data.csv')
        ]

        # Find first existing path
        metadata_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_path = path
                print("path is ", metadata_path)
                break

        # Verify metadata file exists somewhere
        assert metadata_path is not None, f"Metadata file not found in any of these locations: {possible_paths}"

        print ("path is ", metadata_path)
        # Get metadata config for field mappings
        metadata_config = {
            'metadata': {
                'video_segments': config['paths']['data']['metadata']['video_segments']
            }
        }

        # Load metadata with proper field mappings
        metadata = load_video_metadata_file(metadata_path, metadata_config)

        # Verify metadata loaded successfully
        assert metadata is not None, "Failed to load metadata"
        assert len(metadata) > 0, "No entries found in metadata"

        # Print metadata summary for verification
        print(f"\nLoaded metadata from: {metadata_path}")
        print(f"Found {len(metadata)} video entries")
        for i, (video_id, data) in enumerate(list(metadata.items())[:3]):
            print(f"Video {i + 1}: {video_id}")
            if 'repetitions' in data:
                print(f"  Has {len(data['repetitions'])} repetitions")
                for j, rep in enumerate(data['repetitions'][:len(data['repetitions'])]):
                    print(f"  Rep {j + 1}: Frames {rep['start_frame']} to {rep['end_frame']}")
            else:
                print("  No repetitions found")

