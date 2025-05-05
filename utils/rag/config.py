"""
Configuration module for GraphRAG system.

This module handles loading, validating, and accessing configuration settings.
"""

import json
import os
from typing import Any, Dict


class GraphRAGConfig:
    """Configuration class for GraphRAG settings."""

    def __init__(self, config_path: str = "graphrag_config.json"):
        """Initialize configuration from a file.

        Args:
            config_path: Path to the configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Dict containing configuration
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file {self.config_path} not found."
                "Please create a configuration file."
            )

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise RuntimeError(
                f"Error loading config from {self.config_path}: {e}"
            ) from e

    def _save_config(self) -> None:
        """Save the configuration to a file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration to {self.config_path}: {e}")

    def save(self) -> None:
        """Save the current configuration to the config file."""
        self._save_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level configuration value.

        Args:
            key: Configuration key
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing top-level config keys as attributes.

        Args:
            name: Attribute name (config key)

        Returns:
            Configuration value

        Raises:
            AttributeError: If the config key doesn't exist
        """
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'GraphRAGConfig' has no attribute '{name}'")
