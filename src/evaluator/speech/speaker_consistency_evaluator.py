from pdb import run
import pickle

from data_classes.dialogue import Dialogue
from speechbrain.pretrained import SpeakerRecognition
import argparse
import os
import torchaudio
import torch
import numpy as np
import tqdm
from typing import List

from utils.base_classes import SDFModule
from data_classes.evaluation import SpeakerConsistencyEvaluation
import logging

logger = logging.getLogger(__name__)

@SDFModule.set_role("evaluator")
class SpeakerConsistencyEvaluator(SDFModule):

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--speaker_consistency_model_dir",
            type=str,
            default="./pretrained_models/spkrec-xvect-voxceleb",
            help="Path to the speaker consistency model directory",
        )
        parser.add_argument(
            "--speaker_consistency_device",
            type=str,
            default="cuda:0",
            help="Device for the speaker consistency model",
        )
        parser.add_argument(
            "--input_sr",
            type=int,
            default=16000,
            help="Input sample rate for the audio files",
        )
        parser.add_argument(
            "--speaker_consistency_threshold",
            type=float,
            default=0.94,
            help="Threshold for speaker consistency evaluation",
        )

    def __init__(self, args):
        self.args = args
        self.device = args.speaker_consistency_device
        self.input_sr = args.input_sr
        self.threshold = args.speaker_consistency_threshold
    
    def initialize(self):
        savedir = self.args.speaker_consistency_model_dir
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=savedir,
            run_opts={"device": self.device},
        )
        return self


    def evaluate_speaker(self, speaker_idxs, input_sr, audio):
        xs = [
            torchaudio.functional.resample(
                torch.from_numpy(audio[x]).to(self.device), input_sr, 16000
            )
            for x in speaker_idxs[:-1]
        ] + [
            torchaudio.functional.resample(
                torch.from_numpy(audio[speaker_idxs[-1]]).to(self.device),
                input_sr,
                16000,
            )
        ]

        ys = [
            torchaudio.functional.resample(
                torch.from_numpy(audio[x]).to(self.device), input_sr, 16000
            )
            for x in speaker_idxs[1:]
        ] + [
            torchaudio.functional.resample(
                torch.from_numpy(audio[speaker_idxs[0]]).to(self.device),
                input_sr,
                16000,
            )
        ]

        # Update the above code to form a loop i.e. compare the last one with the first one

        xls = np.array([x.shape[0] for x in xs])
        xls = xls / xls.max()
        xls = torch.from_numpy(xls).to(self.device)

        yls = np.array([x.shape[0] for x in ys])
        yls = yls / yls.max()
        yls = torch.from_numpy(yls).to(self.device)

        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)

        with torch.no_grad():
            scores, predictions = self.model.verify_batch(
                xs, ys, xls, yls, threshold=self.threshold
            )
            scores = scores.view(-1).cpu().numpy()
            predictions = predictions.view(-1).cpu().numpy()
        return scores, predictions

    def evaluate_one_conversation(self, dialogue):
        utterances = dialogue.conversation.utterances
        audio = dialogue.dialogue_audio["waveforms"]
        input_sr = dialogue.dialogue_audio["sample_rate"]
        if len(utterances) < 4:
            logger.warning(
                f"Speaker consistency evaluation requires at least 4 utterances. Skipping this dialogue."
            )
            return None
        assert len(audio) == len(utterances)
        s1 = utterances[0].speaker_id
        s2 = utterances[1].speaker_id
        if s1 == s2:
            logger.warning(
                f"Speaker consistency evaluation requires at least 2 different speakers. Skipping this dialogue."
            )
            return None
        s1_utts = list(filter(lambda x: x[1].speaker_id == s1, enumerate(utterances)))
        s2_utts = list(filter(lambda x: x[1].speaker_id == s2, enumerate(utterances)))
        s1_idxs = list(map(lambda x: x[0], s1_utts))
        s2_idxs = list(map(lambda x: x[0], s2_utts))
        #
        # Check s1 first

        s1_scores, s1_predictions = self.evaluate_speaker(s1_idxs, input_sr, audio)
        s2_scores, s2_predictions = self.evaluate_speaker(s2_idxs, input_sr, audio)
        results = {
            "s1_idxs": s1_idxs,
            "s2_idxs": s2_idxs,
            "s1_scores": s1_scores.tolist(),
            "s1_predictions": s1_predictions.tolist(),
            "s2_scores": s2_scores.tolist(),
            "s2_predictions": s2_predictions.tolist(),
        }

        eval_results = SpeakerConsistencyEvaluation(
            overall_speaker_consistency_score=(s1_scores.mean() + s2_scores.mean()) / 2,
            utterance_speaker_consistency_scores=results,
        )
        return eval_results

    def evaluate(self, dialogues: List[Dialogue]):
        for i in tqdm.tqdm(range(len(dialogues))):
            result = self.evaluate_one_conversation(dialogues[i])
            if result is None:
                logger.warning(
                    f"Speaker consistency evaluation failed for dialogue {i}. Skipping this dialogue."
                )
                continue
            dialogues[i].speaker_consistency_evaluation = result
        return dialogues