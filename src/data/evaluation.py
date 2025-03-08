from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import statistics
from data.common import DataClassModel
import pickle


class ScenarioMetadataConsistency(DataClassModel):
    """Evaluates consistency between metadata and user input scenario"""

    dialogue_type_consistency: float = Field(
        ...,
        description="Consistency between dialogue type in metadata and user-specified dialogue type",
    )

    temporal_spatial_consistency: float = Field(
        ...,
        description="Consistency between temporal/spatial setting in metadata and user specifications",
    )

    cultural_background_consistency: float = Field(
        ...,
        description="Consistency between cultural elements in metadata and user-specified cultural background",
    )

    language_norm_consistency: float = Field(
        ...,
        description="Consistency between language usage in metadata and user-specified language",
    )

    custom_prompt_adherence: float = Field(
        ..., description="Adherence of metadata to user's custom prompt requirements"
    )


class MetadataInternalConsistency(DataClassModel):
    """Evaluates internal logical consistency within metadata"""

    character_setting_consistency: float = Field(
        ..., description="Internal consistency of character attributes"
    )

    relationship_logic_consistency: float = Field(
        ..., description="Logical consistency of relationships between characters"
    )

    scene_dialogue_type_consistency: float = Field(
        ..., description="Alignment between scene, time, and dialogue type"
    )

    emotional_tone_consistency: float = Field(
        ..., description="Appropriateness of emotional tone given the scenario"
    )


class MetadataScriptConsistency(DataClassModel):
    """Evaluates consistency between metadata and script"""

    character_personality_alignment: float = Field(
        ...,
        description="Alignment between character traits in metadata and behavior in script",
    )

    relationship_dynamic_alignment: float = Field(
        ...,
        description="Alignment between relationship dynamic in metadata and interactions in script",
    )

    setting_alignment: float = Field(
        ..., description="Alignment between setting description in metadata and script"
    )

    topic_goal_alignment: float = Field(
        ...,
        description="Alignment between main topic in metadata and narrative focus in script",
    )


class ScriptDialogueConsistency(DataClassModel):
    """Evaluates consistency between script and dialogue"""

    narrative_structure_adherence: float = Field(
        ..., description="How well dialogue follows the script's narrative structure"
    )

    key_points_coverage: float = Field(
        ..., description="Coverage of key points from script in dialogue"
    )

    emotional_progression_alignment: float = Field(
        ...,
        description="Alignment between emotional progression in script and dialogue",
    )

    character_behavior_alignment: float = Field(
        ...,
        description="Alignment between character behaviors described in script and actual dialogue",
    )


class MetadataDialogueConsistency(DataClassModel):
    """Evaluates direct consistency between metadata and dialogue"""

    character_background_reflection: float = Field(
        ...,
        description="How well dialogue reflects character backgrounds from metadata",
    )

    setting_details_reflection: float = Field(
        ..., description="How well dialogue incorporates setting details from metadata"
    )

    language_style_alignment: float = Field(
        ..., description="Alignment between language style in metadata and dialogue"
    )

    topic_focus_alignment: float = Field(
        ...,
        description="Alignment between main topic in metadata and actual dialogue content",
    )


class CrossComponentConsistency(DataClassModel):
    """Evaluates consistency across metadata, script, and dialogue"""

    metadata_script_consistency: MetadataScriptConsistency = Field(
        ..., description="Consistency between metadata and script"
    )

    script_dialogue_consistency: ScriptDialogueConsistency = Field(
        ..., description="Consistency between script and dialogue"
    )

    metadata_dialogue_consistency: MetadataDialogueConsistency = Field(
        ..., description="Direct consistency between metadata and dialogue"
    )


class ConsistencyEvaluation(DataClassModel):
    """Complete consistency evaluation across all components"""

    scenario_metadata_consistency: ScenarioMetadataConsistency = Field(
        ..., description="Consistency between scenario and metadata"
    )

    metadata_internal_consistency: MetadataInternalConsistency = Field(
        ..., description="Internal consistency within metadata"
    )

    cross_component_consistency: CrossComponentConsistency = Field(
        ..., description="Consistency across metadata, script, and dialogue"
    )

    overall_consistency_score: float = Field(
        ..., description="Overall consistency score across all dimensions"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsistencyEvaluation":
        """Create a model from a dictionary"""
        return cls.parse_obj(data)

    def save_to_json(self, file_path: str, pretty: bool = True) -> None:
        """Save the ConsistencyEvaluation to a JSON file.

        Args:
            file_path: Path to save the JSON file.
            pretty: If True, format the JSON with indentation for readability.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.json(indent=4 if pretty else None))

    @classmethod
    def load_from_json(cls, file_path: str) -> "ConsistencyEvaluation":
        """Load a ConsistencyEvaluation from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            A new ConsistencyEvaluation instance.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.parse_raw(f.read())

    def save_to_pickle(self, file_path: str) -> None:
        """Save the ConsistencyEvaluation to a pickle file.

        Args:
            file_path: Path to save the pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, file_path: str) -> "ConsistencyEvaluation":
        """Load a ConsistencyEvaluation from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            A new ConsistencyEvaluation instance.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save_batch_to_pickle(
        cls, evaluations: List["ConsistencyEvaluation"], file_path: str
    ) -> None:
        """Save a batch of ConsistencyEvaluations to a pickle file.

        Args:
            evaluations: List of ConsistencyEvaluation instances to save.
            file_path: Path to save the pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(evaluations, f)

    @classmethod
    def load_batch_from_pickle(cls, file_path: str) -> List["ConsistencyEvaluation"]:
        """Load a batch of ConsistencyEvaluations from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            List of ConsistencyEvaluation instances.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)


