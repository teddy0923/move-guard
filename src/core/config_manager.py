# core/config_manager.py
import yaml
import os
import logging
from pathlib import Path


class ConfigLoader:
    """Utility for loading and managing configuration from YAML files"""

    def __init__(self, config_dir="config"):
        """
        Initialize the config loader

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.configs = {}

    def load_config(self, config_name):
        """
        Load a specific configuration file

        Args:
            config_name: Name of the configuration file (without .yaml extension)

        Returns:
            config: Dictionary with configuration parameters
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")

        if not os.path.exists(config_path):
            logging.error(f"Configuration file not found: {config_path}")
            return None

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.configs[config_name] = config
                return config
        except Exception as e:
            logging.error(f"Error loading configuration file {config_path}: {str(e)}")
            return None

    def merge_configs(self, base_config_name, specific_config_name):
        """
        Merge a specific configuration with a base configuration

        Args:
            base_config_name: Name of the base configuration
            specific_config_name: Name of the specific configuration to merge

        Returns:
            merged_config: Dictionary with merged configuration parameters
        """
        base_config = self.configs.get(base_config_name)
        if base_config is None:
            base_config = self.load_config(base_config_name)
            if base_config is None:
                return None

        specific_config = self.configs.get(specific_config_name)
        if specific_config is None:
            specific_config = self.load_config(specific_config_name)
            if specific_config is None:
                return None

        # Create a deep copy of the base config
        import copy
        merged_config = copy.deepcopy(base_config)

        # Recursively merge the configs
        self._recursive_merge(merged_config, specific_config)

        return merged_config

    def _recursive_merge(self, base, specific):
        """
        Recursively merge two dictionaries

        Args:
            base: Base dictionary to merge into
            specific: Specific dictionary to merge from
        """
        for key, value in specific.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._recursive_merge(base[key], value)
            else:
                base[key] = value