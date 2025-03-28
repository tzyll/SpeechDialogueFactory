from data_classes import dialogue
from utils.base_classes import SDFModule
import whisper
import argparse
import pandas as pd
import jiwer
import tqdm
import librosa
import sys
import os
import pickle
from data_classes.dialogue import Dialogue
from data_classes.evaluation import IntelligibilityEvaluation


wer_transform_en = jiwer.Compose(
    [
        jiwer.RemoveKaldiNonWords(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
wer_transform_zh = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

@SDFModule.set_role("evaluator")
class IntelligibilityEvaluator(SDFModule):

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--whisper_model_name",
            type=str,
            default="base",
            help="Name of the Whisper model to use for evaluation.",
        )
        parser.add_argument(
            "--whisper_device",
            type=str,
            default="cuda:0",
            help="Device to run the Whisper model on.",
        )
        parser.add_argument(
            "--whisper_input_sr",
            type=int,
            default=16000,
            help="Input sample rate for the Whisper model.",
        )
        parser.add_argument(
            "--input_dialogue_file",
            type=str,
            default="./dialogue.pkl",
            help="Path to the input dialogue file",
        )
        parser.add_argument(
            "--output_dialogue_file",
            type=str,
            default="./dialogue.pkl",
            help="Path to the output dialogue file",
        )

    def __init__(self, args):
        self.args = args
        self.input_sr = args.whisper_input_sr
        self.language_transformation = {
            "English": wer_transform_en,
            "Chinese": wer_transform_en,
        }
        self.error_rate_function = {
            "English": jiwer.wer,
            "Chinese": jiwer.cer,
        }
        self.initialize()
        self.input_dialogue_file = args.input_dialogue_file
        self.output_dialogue_file = args.output_dialogue_file

    def initialize(self):
        whisper_model_name = self.args.whisper_model_name
        whisper_device = self.args.whisper_device
        self.model = whisper.load_model(whisper_model_name)
        self.model.eval().to(whisper_device)
        return self


    def evaluate_one_dialogue(self, dialogue: Dialogue):
        audio_data = dialogue.dialogue_audio["waveforms"]
        input_sr = dialogue.dialogue_audio["sample_rate"]
        texts = []
        assert len(audio_data) == len(dialogue.conversation.utterances)
        dialogue_language = dialogue.scenario.dialogue_language
        for u in audio_data:
            wave = librosa.resample(u, orig_sr=input_sr, target_sr=16000)
            result = self.model.transcribe(audio=wave, language=dialogue_language)
            texts.append(result["text"])
        references = [d.text for d in dialogue.conversation.utterances]
        if len(references) != len(texts):
            print("Warning: number of references and transcriptions do not match")
            return {
                "dialogue_wer": 1.0,
                "utterance_wers": [],
            }
        wer_transform = self.language_transformation[dialogue_language]
        error_rate_function = self.error_rate_function[dialogue_language]
        dialogue_wer = error_rate_function(
            references,
            texts,
            truth_transform=wer_transform,
            hypothesis_transform=wer_transform,
        )

        utterance_wers = [
            error_rate_function(
                r, t, truth_transform=wer_transform, hypothesis_transform=wer_transform
            )
            for r, t in zip(references, texts)
        ]
        return {
            "dialogue_wer": dialogue_wer,
            "utterance_wers": utterance_wers,
        }

    def evaluate(
        self,
    ):
        """Evaluate the intelligibility of a list of dialogues.

        Args:
            dialogues: List of Dialogue instances to evaluate.
            audio_files: List of paths to audio files corresponding to the dialogues.

        Returns:
            A list of dictionaries containing evaluation results for each dialogue.
        """
        dialogues = Dialogue.load_batch_from_pickle(self.input_dialogue_file)
        for i in tqdm.tqdm(range(len(dialogues))):
            result = self.evaluate_one_dialogue(dialogues[i])
            intelligibility_evaluation = IntelligibilityEvaluation(
                dialogue_wer=result["dialogue_wer"],
                utterance_wers=result["utterance_wers"],
            )
            dialogues[i].intelligibility_evaluation = intelligibility_evaluation
        Dialogue.save_batch_to_pickle(
            dialogues, self.output_dialogue_file
        )
        return dialogues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    IntelligibilityEvaluator.add_arguments(parser)
    args = parser.parse_args()
    evaluator = IntelligibilityEvaluator(args)
    evaluator.evaluate()