from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional

from sympy import O, Ordinal
from data_classes.common import DataClassModel
import pickle
import numpy as np


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

    scenario_metadata_consistency_score: Optional[float] = Field(
        default=None, description="Overall score for scenario metadata consistency"
    )
    metadata_internal_consistency_score: Optional[float] = Field(
        default=None,
        description="Overall score for internal consistency within metadata",
    )
    cross_component_consistency_score: Optional[float] = Field(
        default=None, description="Overall score for cross-component consistency"
    )

    overall_consistency_score: float = Field(
        ..., description="Overall consistency score across all dimensions"
    )

    model_config = {
        "json_schema_extra": {
            "exclude": [
                "scenario_metadata_consistency_score",
                "metadata_internal_consistency_score",
                "cross_component_consistency_score",
            ]
        }
    }

    def summary(self):
        # Return high-level scores instead of detailed attributes
        return {
            "scenario_metadata_consistency_score": self.scenario_metadata_consistency_score,
            "metadata_internal_consistency_score": self.metadata_internal_consistency_score,
            "cross_component_consistency_score": self.cross_component_consistency_score,
            "overall_consistency_score": self.overall_consistency_score,
        }


class TurnCoherence(DataClassModel):
    """Evaluates coherence of dialogue"""

    turn_id: Optional[int] = Field(
        None, description="ID of the dialogue turn being evaluated"
    )
    topic_relevance: float = Field(
        ..., description="Relevance of the dialogue to the main topic"
    )

    contextual_follow_up: float = Field(
        ..., description="Contextual relevance of follow-up responses"
    )

    logical_continuity: float = Field(
        ..., description="Logical flow and continuity in the dialogue"
    )

    no_contradiction: float = Field(
        ..., description="Absence of contradictions in the dialogue"
    )

    coherence_score: float = Field(
        ..., description="Overall coherence score based on the above factors"
    )


class CoherenceEvaluation(DataClassModel):
    turns_coherence: List[TurnCoherence] = Field(
        ..., description="List of coherence evaluations for each dialogue turn"
    )
    topic_relevance_score: float = Field(
        default=None,
        description="Overall score for topic relevance across the entire dialogue",
    )
    contextual_follow_up_score: Optional[float] = Field(
        default=None,
        description="Overall score for contextual follow-up across the entire dialogue",
    )
    logical_continuity_score: Optional[float] = Field(
        default=None,
        description="Overall score for logical continuity across the entire dialogue",
    )
    no_contradiction_score: Optional[float] = Field(
        default=None,
        description="Overall score for absence of contradictions across the entire dialogue",
    )
    overall_coherence_score: Optional[float] = Field(
        default=None, description="Overall coherence score across the entire dialogue"
    )

    model_config = {
        "json_schema_extra": {
            "exclude": [
                "topic_relevance_score",
                "contextual_follow_up_score",
                "logical_continuity_score",
                "no_contradiction_score",
                "overall_coherence_score",
            ]
        }
    }

    def summary(self):
        # Return high-level scores instead of detailed attributes
        return {
            "topic_relevance_score": self.topic_relevance_score,
            "contextual_follow_up_score": self.contextual_follow_up_score,
            "logical_continuity_score": self.logical_continuity_score,
            "no_contradiction_score": self.no_contradiction_score,
            "overall_coherence_score": self.overall_coherence_score,
        }


class TurnNaturalness(DataClassModel):
    """Evaluates naturalness of dialogue"""

    turn_id: Optional[int] = Field(
        None, description="ID of the dialogue turn being evaluated"
    )
    oral_style: float = Field(
        ..., description="Naturalness of oral style in the dialogue"
    )

    length_and_flow: float = Field(
        ..., description="Naturalness of length and flow in the dialogue"
    )

    emotion_appropriateness: float = Field(
        ..., description="Appropriateness of emotional expression in the dialogue"
    )

    text_emotion_consistency: float = Field(
        ..., description="Consistency between text and emotional tone"
    )

    contextual_vocabulary_style: float = Field(
        ..., description="Naturalness of vocabulary and style in context"
    )

    naturalness_score: float = Field(
        ..., description="Overall naturalness score based on the above factors"
    )


class NaturalnessEvaluation(DataClassModel):
    """Evaluates naturalness of dialogue"""

    turns_naturalness: List[TurnNaturalness] = Field(
        ..., description="List of naturalness evaluations for each dialogue turn"
    )

    oral_style_score: Optional[float] = Field(
        default=None,
        description="Overall score for oral style across the entire dialogue",
    )
    length_and_flow_score: Optional[float] = Field(
        default=None,
        description="Overall score for length and flow across the entire dialogue",
    )
    emotion_appropriateness_score: Optional[float] = Field(
        default=None,
        description="Overall score for emotional appropriateness across the entire dialogue",
    )
    text_emotion_consistency_score: Optional[float] = Field(
        default=None,
        description="Overall score for text-emotion consistency across the entire dialogue",
    )
    contextual_vocabulary_style_score: Optional[float] = Field(
        default=None,
        description="Overall score for contextual vocabulary and style across the entire dialogue",
    )
    overall_naturalness_score: Optional[float] = Field(
        default=None, description="Overall naturalness score across the entire dialogue"
    )
    model_config = {
        "json_schema_extra": {
            "exclude": [
                "oral_style_score",
                "length_and_flow_score",
                "emotion_appropriateness_score",
                "text_emotion_consistency_score",
                "contextual_vocabulary_style_score",
                "overall_naturalness_score",
            ]
        }
    }

    def summary(self):
        # Return high-level scores instead of detailed attributes
        return {
            "oral_style_score": self.oral_style_score,
            "length_and_flow_score": self.length_and_flow_score,
            "emotion_appropriateness_score": self.emotion_appropriateness_score,
            "text_emotion_consistency_score": self.text_emotion_consistency_score,
            "contextual_vocabulary_style_score": self.contextual_vocabulary_style_score,
            "overall_naturalness_score": self.overall_naturalness_score,
        }


class IntelligibilityEvaluation(DataClassModel):
    """Evaluates intelligibility of dialogue"""

    dialogue_wer: float = Field(
        ..., description="Word Error Rate (WER) for the entire dialogue"
    )

    utterance_wers: List[float] = Field(
        ..., description="List of WERs for each utterance in the dialogue"
    )

    def summary(self):
        results = {
            f"turn_{i}": self.utterance_wers[i] for i in range(len(self.utterance_wers))
        }
        results["overall_wer"] = self.dialogue_wer
        return results


class SpeechQualityEvaluation(DataClassModel):
    """Evaluates speech quality of dialogue"""

    mos: float = Field(
        ..., description="Mean Opinion Score (MOS) for the entire dialogue"
    )
    production_quality: float = Field(
        ..., description="Overall production quality score for the dialogue"
    )
    production_complexity: float = Field(
        ..., description="Overall production complexity score for the dialogue"
    )
    content_enjoyment: float = Field(
        ..., description="Overall content enjoyment score for the dialogue"
    )
    content_usefulness: float = Field(
        ..., description="Overall content usefulness score for the dialogue"
    )
    utterance_quality_scores: List[Dict[str, Any]] = Field(
        ..., description="List of quality scores for each utterance in the dialogue"
    )

    def summary(self):
        # Return high-level scores instead of detailed attributes
        return {
            "mos": self.mos,
            "production_quality": self.production_quality,
            "production_complexity": self.production_complexity,
            "content_enjoyment": self.content_enjoyment,
            "content_usefulness": self.content_usefulness,
        }


class SpeakerConsistencyEvaluation(DataClassModel):
    """Evaluates speaker consistency in dialogue"""

    overall_speaker_consistency_score: float = Field(
        ..., description="Overall speaker consistency score for the dialogue"
    )

    utterance_speaker_consistency_scores: Dict[str, Any] = Field(
        ..., description="Dictionary of speaker consistency scores for each utterance"
    )

    def summary(self):
        # Return high-level scores instead of detailed attributes

        results = {
            "speaker_1_consistency": float(np.mean(
                self.utterance_speaker_consistency_scores["s1_scores"]
            )),
            "speaker_2_consistency": float(np.mean(
                self.utterance_speaker_consistency_scores["s2_scores"]
            )),
        }
        return results
