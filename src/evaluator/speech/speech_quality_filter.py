from venv import logger
from utils.base_classes import SDFModule
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue
from data_classes.evaluation import (
    IntelligibilityEvaluation,
    SpeakerConsistencyEvaluation,
    SpeechQualityEvaluation,
)
import numpy as np

logger = logging.getLogger(__name__)


@SDFModule.set_role("evaluator")
class SpeechQualityFilter(SDFModule):
    def __init__(self, args, llm: LLM = None):
        self.llm = llm
        self.args = args
        self.speech_quality_threshold = args.speech_quality_threshold
        self.intelligibility_threshold = args.intelligibility_threshold
        self.speaker_consistency_threshold = args.speaker_consistency_threshold

    def _is_valid(self, dialogue: Dialogue) -> bool:
        """
        Check if the dialogue passes the speech quality filter.

        Args:
            dialogue (Dialogue): Dialogue object to be checked.

        Returns:
            bool: True if the dialogue passes the filter, False otherwise.
        """
        # speech_quality_score = np.mean(dialogue.speech_quality_evaluation.summary().values())
        speech_quality_scores = dialogue.speech_quality_evaluation.summary()
        speech_quality_score = np.mean(
            list({
                "mos": speech_quality_scores["mos"] * 2 / 10,
                # "production_quality": speech_quality_scores["production_quality"] / 10,
                # "production_complexity": (
                #     10 - speech_quality_scores["production_complexity"]
                # )
                # / 10,
                # "content_enjoyment": speech_quality_scores["content_enjoyment"] / 10,
                # "content_usefulness": speech_quality_scores["content_usefulness"] / 10,
            }.values())
        )
        intelligibility_score = (
            1 - dialogue.intelligibility_evaluation.summary()["overall_wer"]
        )
        speaker_consistency_score = np.mean(
            list(dialogue.speaker_consistency_evaluation.summary().values())
        )
        logger.info(
            f"Speech Quality score: {speech_quality_score}, Intelligibility score: {intelligibility_score}, Speaker Consistency score: {speaker_consistency_score}"
        )
        if (
            speech_quality_score >= self.speech_quality_threshold
            and intelligibility_score >= self.intelligibility_threshold
            and speaker_consistency_score >= self.speaker_consistency_threshold
        ):
            return True
        else:
            return False

    def evaluate(
        self,
        dialogues: List[Dialogue],
    ) -> List[Dialogue]:
        """
        Evaluate the quality of dialogues based on speech quality, intelligibility, and speaker consistency.

        Args:
            dialogues (List[Dialogue]): List of Dialogue objects to be evaluated.

        Returns:
            List[Dialogue]: List of Dialogue objects that pass the quality filter.
        """
        logger.info("Evaluating speech quality of dialogues...")
        filtered_dialogues = []
        for dialogue in dialogues:
            if self._is_valid(dialogue):
                filtered_dialogues.append(dialogue)
        logger.info(
            f"Filtered dialogues: {len(filtered_dialogues)} out of {len(dialogues)}"
        )
        return filtered_dialogues