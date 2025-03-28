from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import json
import pickle


class DataClassModel(BaseModel):
    """Base class for all models in this module."""

    def to_json(self, pretty: bool = False) -> str:
        """Convert the model to a JSON string."""
        if pretty:
            return self.model_dump_json(indent=2)
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str):
        """Create a Dialogue instance from a JSON string.

        Args:
            json_str: JSON string representation of a Dialogue.

        Returns:
            A new Dialogue instance.
        """
        return cls.model_validate_json(json_str)

    def to_dict(self):
        """Convert the model to a dictionary."""
        return self.model_dump()

    def from_dict(cls, data):
        """Create a model from a dictionary."""
        return cls.model_validate(data)

    def save_to_json(self, file_path: str, pretty: bool = False) -> None:
        """Save the model to a JSON file.

        Args:
            file_path: Path to save the JSON file.
            pretty: Whether to format the JSON output.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(pretty=pretty))

    @classmethod
    def load_from_json(cls, file_path: str) -> "DataClassModel":
        """Load a model from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            A new model instance.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def save_to_pickle(self, file_path: str) -> None:
        """Save the model to a pickle file.

        Args:
            file_path: Path to save the pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, file_path: str) -> "DataClassModel":
        """Load a model from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            A new model instance.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save_batch_to_pickle(
        cls, models: List["DataClassModel"], file_path: str
    ) -> None:
        """Save a batch of models to a pickle file.

        Args:
            models: List of model instances to save.
            file_path: Path to save the pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(models, f)

    @classmethod
    def load_batch_from_pickle(cls, file_path: str) -> List["DataClassModel"]:
        """Load a batch of models from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            List of model instances.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save_batch_to_json(
        cls, models: List["DataClassModel"], file_path: str, pretty: bool = False
    ) -> None:
        """Save a batch of models to a JSON file.

        Args:
            models: List of model instances to save.
            file_path: Path to save the JSON file.
            pretty: Whether to format the JSON output.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for model in models:
                f.write(model.to_json(pretty=pretty) + "\n")

    @classmethod
    def load_batch_from_json(cls, file_path: str) -> List["DataClassModel"]:
        """Load a batch of models from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of model instances.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return [cls.from_json(line) for line in f]


    def summary(self) -> str:
        """
        Return high-level summary of the model's attributes.

        """
        return self.to_dict()