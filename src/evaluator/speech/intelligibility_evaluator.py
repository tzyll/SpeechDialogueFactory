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
import subprocess
import time
import logging
from data_classes.dialogue import Dialogue
from data_classes.evaluation import IntelligibilityEvaluation
import torch

logger = logging.getLogger(__name__)

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
            "--num_whisper_workers",
            type=int,
            default=4,
            help="Number of workers for processing audio files",
        )
        parser.add_argument(
            "--intelligibility_evaluation_temp_dir",
            type=str,
            default="./tmp_evaluation",
            help="Temporary directory for storing intermediate evaluation files",
        )

    def __init__(self, args):
        self.args = args
        self.input_sr = args.whisper_input_sr
        self.num_workers = args.num_whisper_workers
        self.whisper_model_name = args.whisper_model_name
        self.evaluation_temp_dir = os.path.abspath(args.intelligibility_evaluation_temp_dir)
        os.makedirs(self.evaluation_temp_dir, exist_ok=True)
        
    def initialize(self):
        return self

    def evaluate(
        self,
        dialogues: list[Dialogue],
    ):
        # Get number of available GPUs
        n_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {n_gpus}")

        # Calculate number of dialogues and processes
        num_dialogues = len(dialogues)
        logger.info(f"Number of dialogues to evaluate: {num_dialogues}")
        total_num_processes = self.num_workers * n_gpus
        logger.info(f"Total number of processes: {total_num_processes}")
        if total_num_processes > num_dialogues:
            total_num_processes = num_dialogues
            logger.info(f"Adjusted total number of processes to {total_num_processes} based on number of dialogues")
        logger.info(f"Will use {total_num_processes} processes for evaluation")

        # Split dialogues into chunks
        chunk_size = max(1, num_dialogues // total_num_processes + 1)
        dialogue_chunks = [
            dialogues[i : i + chunk_size] for i in range(0, num_dialogues, chunk_size)
        ]
        logger.info(f"Split dialogues into {len(dialogue_chunks)} chunks")

        # Create temp dialogue shards
        dialogue_shards = []
        for i, chunk in enumerate(dialogue_chunks):
            dialogue_shard = os.path.join(self.evaluation_temp_dir, f"input_dialogue_{i}.pkl")
            Dialogue.save_batch_to_pickle(chunk, dialogue_shard)
            dialogue_shards.append(dialogue_shard)

        # Get the path of this file
        current_path = os.path.dirname(os.path.abspath(__file__))

        # Call worker script in subprocesses
        commands = []
        output_files = []
        for i, dialogue_shard in enumerate(dialogue_shards):
            output_file_name = os.path.join(self.evaluation_temp_dir, f"output_dialogue_{i}.pkl")
            output_files.append(output_file_name)
            cmd = [
                "python",
                f"{current_path}/intelligibility_evaluator_worker.py",
                "--whisper_model_name", self.whisper_model_name,
                "--input_dialogue_file", dialogue_shard,
                "--output_dialogue_file", output_file_name,
                "--whisper_input_sr", str(self.input_sr)
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpus)
            commands.append((cmd, env))

        logger.info(f"Prepared {len(commands)} commands for evaluation")

        # Launch processes
        processes = [subprocess.Popen(cmd, env=env) for cmd, env in commands]
        logger.info(f"Started {len(processes)} processes for intelligibility evaluation")
        for p in processes:
            logger.info(f"Started process with PID: {p.pid}")

        # Monitor processes until completion
        while processes:
            for p in processes[:]:
                retcode = p.poll()
                if retcode is not None:  # Process has finished
                    logger.info(f"Process {p.pid} finished with return code {retcode}")
                    processes.remove(p)
            time.sleep(5)  # Check process status every 5 seconds

        # Read the temporary files and combine results
        final_dialogues = []
        for output_file in output_files:
            dialogues_chunk = Dialogue.load_batch_from_pickle(output_file)
            final_dialogues.extend(dialogues_chunk)

        # Clean up temporary files
        for dialogue_shard in dialogue_shards:
            os.remove(dialogue_shard)
        for output_file in output_files:
            os.remove(output_file)

        # delete the temporary directory
        os.rmdir(self.evaluation_temp_dir)
        logger.info(f"Deleted temporary directory {self.evaluation_temp_dir}")

        return final_dialogues
