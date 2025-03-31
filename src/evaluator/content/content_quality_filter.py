from venv import logger
from utils.base_classes import SDFModule
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue
from data_classes.evaluation import (
    ConsistencyEvaluation,
    CoherenceEvaluation,
    NaturalnessEvaluation,
)
import numpy as np

logger = logging.getLogger(__name__)


@SDFModule.set_role("evaluator")
class ContentQualityFilter(SDFModule):
    def __init__(self, args, llm: LLM = None):
        self.llm = llm
        self.args = args
        self.consistency_threshold = args.consistency_threshold
        self.coherence_threshold = args.coherence_threshold
        self.naturalness_threshold = args.naturalness_threshold

    def _is_valid(self, dialogue: Dialogue) -> bool:
        """
        Check if the dialogue passes the content quality filter.

        Args:
            dialogue (Dialogue): Dialogue object to be checked.

        Returns:
            bool: True if the dialogue passes the filter, False otherwise.
        """
        if len(dialogue.conversation.utterances) < 4:
            return False
        coherence_score = np.mean(
            list(dialogue.coherence_evaluation.summary().values())
        )
        consistency_score = np.mean(
            list(dialogue.consistency_evaluation.summary().values())
        )
        naturalness_score = np.mean(
            list(dialogue.naturalness_evaluation.summary().values())
        )
        logger.info(
            f"Coherence score: {coherence_score}, Consistency score: {consistency_score}, Naturalness score: {naturalness_score}"
        )
        if (
            coherence_score >= self.coherence_threshold
            and consistency_score >= self.consistency_threshold
            and naturalness_score >= self.naturalness_threshold
        ):
            return True
        else:
            return False

    def evaluate(
        self,
        dialogues: List[Dialogue],
    ) -> List[Dialogue]:
        """
        Evaluate the quality of dialogues based on coherence, consistency, and naturalness.

        Args:
            dialogues (List[Dialogue]): List of Dialogue objects to be evaluated.

        Returns:
            List[Dialogue]: List of Dialogue objects that pass the quality filter.
        """

        logger.info("Evaluating content quality of dialogues...")
        filtered_dialogues = []
        for dialogue in dialogues:
            if self._is_valid(dialogue):
                filtered_dialogues.append(dialogue)
        logger.info(
            f"Filtered dialogues: {len(filtered_dialogues)} out of {len(dialogues)}"
        )
        return filtered_dialogues
