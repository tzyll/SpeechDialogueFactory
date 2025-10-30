from tkinter import dialog
import torch
import argparse
import pickle
import pandas as pd
from zmq import device
from data_classes.evaluation import SpeechQualityEvaluation
import librosa
from utils.base_classes import SDFModule
import utmosv2
import os
import tqdm
import itertools
import soundfile as sf
from audiobox_aesthetics.infer import initialize_predictor
import numpy as np
from data_classes.dialogue import Dialogue
import gc


@SDFModule.set_role("evaluator")
class SpeechQualityEvaluator(SDFModule):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model_path",
            type=str,
            default="./third_parties/pretrained_models/UTMOSv2/fusion_stage3/fold0_s42_best_model.pth",
            help="Path to the UTMOSv2 model checkpoint",
        )
        parser.add_argument(
            "--input_sr",
            type=int,
            default=16000,
            help="Input sample rate for the audio files",
        )
        parser.add_argument(
            "--mos_tmp_dir",
            type=str,
            default="./tmp_mos_files",
            help="Temporary directory for storing audio files",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers for processing audio files",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size for processing audio files",
        )
        parser.add_argument(
            "--mos_device",
            type=str,
            default="cuda:0",
            help="Device for the MOS model",
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
            default="./dialogue_with_quality.pkl",
            help="Path to the output dialogue file",
        )

    def __init__(self, args):
        self.args = args
        self.device = args.mos_device
        self.input_sr = args.input_sr
        self.tmp_dir = args.mos_tmp_dir
        self.tmp_dir = os.path.abspath(self.tmp_dir)
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.input_dialogue_file = args.input_dialogue_file
        self.output_dialogue_file = args.output_dialogue_file
        self.initialize()

    def initialize(self):
        self.mos_model = utmosv2.create_model(checkpoint_path=self.args.model_path, device=self.device)
        self.aesthetics_model = initialize_predictor()
        return self

    def preprocess_audio(self, dialogues):
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        else:
            for file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, file))
        total = len(
            list(
                itertools.chain.from_iterable(
                    [dialogue.conversation.utterances for dialogue in dialogues]
                )
            )
        )
        all_audio_files = []
        with tqdm.tqdm(total=total) as pbar:
            for i, dialogue in enumerate(dialogues):
                audio_data = dialogue.dialogue_audio["waveforms"]
                input_sr = dialogue.dialogue_audio["sample_rate"]
                assert len(audio_data) == len(dialogue.conversation.utterances)
                for j, waveform in enumerate(audio_data):
                    wave = librosa.resample(waveform, orig_sr=input_sr, target_sr=16000)
                    file_name = os.path.join(self.tmp_dir, f"mos_{i}_{j}.wav")
                    sf.write(file_name, wave, 16000)
                    all_audio_files.append(file_name)
                    pbar.update(1)
        return all_audio_files

    def evaluate_aesthetics(self, audio_files):
        # Run in a batch manner by iterating over the audio files, pass a list of file paths into the model
        qualities = []
        for i in tqdm.tqdm(range(0, len(audio_files), self.batch_size)):
            batch_files = audio_files[i : i + self.batch_size]
            model_inputs = [{"path": x} for x in batch_files]
            aes_scores = self.aesthetics_model.forward(model_inputs)
            qualities.extend(aes_scores)
        # Convert to a dict with file name as key and quality score as value
        qualities = {os.path.basename(x): y for x, y in zip(audio_files, qualities)}
        del self.aesthetics_model
        gc.collect()
        torch.cuda.empty_cache()
        return qualities

    def evaluate(self):
        dialogues = Dialogue.load_batch_from_pickle(self.input_dialogue_file)
        all_audio_files = self.preprocess_audio(dialogues)
        quality_scores = self.evaluate_aesthetics(all_audio_files)
        mos = self.mos_model.predict(
            input_dir=self.tmp_dir,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        assert len(mos) == len(
            quality_scores
        ), "Number of MOS scores and quality scores do not match"
        mos_dict = {
            os.path.basename(m["file_path"]): m["predicted_mos"]
            for m in mos
            if m["file_path"] in all_audio_files
        }
        for i in range(len(dialogues)):
            utterance_speech_qualities = []
            for j in range(len(dialogues[i].conversation.utterances)):
                quality_score = quality_scores[f"mos_{i}_{j}.wav"]
                mos_score = mos_dict[f"mos_{i}_{j}.wav"]
                utterance_speech_qualities.append({**quality_score, "MOS": mos_score})
            dialogues[i].speech_quality_evaluation = SpeechQualityEvaluation(
                mos=np.mean([x["MOS"] for x in utterance_speech_qualities]),
                production_quality=np.mean(
                    [x["PQ"] for x in utterance_speech_qualities]
                ),
                production_complexity=np.mean(
                    [x["PC"] for x in utterance_speech_qualities]
                ),
                content_enjoyment=np.mean(
                    [x["CE"] for x in utterance_speech_qualities]
                ),
                content_usefulness=np.mean(
                    [x["CU"] for x in utterance_speech_qualities]
                ),
                utterance_quality_scores=utterance_speech_qualities,
            )
        # Clean up temporary files
        for file in all_audio_files:
            os.remove(file)
        # save the dialogues with the quality scores
        Dialogue.save_batch_to_pickle(
            dialogues, self.output_dialogue_file
        )
        return dialogues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    SpeechQualityEvaluator.add_arguments(parser)
    args = parser.parse_args()
    evaluator = SpeechQualityEvaluator(args)
    evaluator.evaluate()