from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import json
import pickle


class DataClassModel(BaseModel):
    """Base class for all models in this module."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return self.dict()

    def to_json(self, pretty: bool = False) -> str:
        """Convert the model to a JSON string."""
        if pretty:
            return json.dumps(self.dict(), indent=2, ensure_ascii=False)
        return json.dumps(self.dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str):
        """Create a Dialogue instance from a JSON string.

        Args:
            json_str: JSON string representation of a Dialogue.

        Returns:
            A new Dialogue instance.
        """
        data = json.loads(json_str)
        return cls.parse_obj(data)
