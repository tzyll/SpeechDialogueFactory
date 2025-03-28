import os
import sys
import torchaudio
import re
import pandas as pd
import logging
import random
from data_classes.dialogue import Dialogue, Role
import torch
import argparse

from utils.base_classes import SDFModule
import librosa
import time
import subprocess

logger = logging.getLogger(__name__)


@SDFModule.set_role("generator")
class CosyVoiceTTS(SDFModule):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--cosyvoice_codebase",
            type=str,
            default="cosyvoice",
            help="Path to the CosyVoice codebase",
        )
        parser.add_argument(
            "--cosyvoice_model_checkpoint",
            type=str,
            default="cosyvoice/tts",
            help="Path to the CosyVoice model checkpoint",
        )
        parser.add_argument(
            "--cosyvoice_voice_bank_path",
            type=str,
            default="cosyvoice/voice_bank",
            help="Path to the voice bank directory",
        )
        parser.add_argument(
            "--target_sample_rate",
            type=int,
            default=16000,
            help="Target sample rate for the audio output",
        )
        parser.add_argument(
            "--cosyvoice_device",
            type=str,
            default="cuda:0",
            help="Device for the CosyVoice model",
        )
        parser.add_argument(
            "--cosyvoice_tmp_dir",
            type=str,
            default="cosyvoice",
            help="Type of the CosyVoice model",
        )
        parser.add_argument(
            "--num_tts_workers",
            type=int,
            default=4,
            help="Number of workers for TTS generation",
        )

    def __init__(self, args):
        self.args = args
        self.target_sample_rate = args.target_sample_rate
        self.num_tts_workers = args.num_tts_workers

    def initialize(self):
        return self

    def generate(self, dialogues: list[Dialogue], gen_params={}):
        # Here we will call the cosyvoice_tts_worker.py in several subprocesses to parallelize the TTS generation
        # We will first need to create N temporary file to store the dialogue shards base on the number of available GPU
        # We will call the cosyvoice_tts_worker.py with the path to the temporary file and the path to the cosyvoice model as well need to set the device
        # We will need to keep monitorying the processes and wait for them to finish
        # Then we will read the temporary files and concatenate the audio files
        # Finally we will return dialogues with the audio data inside.
        # We will also need to remove the temporary files

        n_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {n_gpus}")
        num_dialogues = len(dialogues)
        logger.info(f"Number of dialogues: {num_dialogues}")
        num_processes_per_gpu = self.num_tts_workers
        total_num_processes = n_gpus * num_processes_per_gpu
        logger.info(f"Total number of processes: {total_num_processes}")

        # Split dialogues into chunks based on the number of processes
        chunk_size = max(1, num_dialogues // total_num_processes + 1)
        dialogue_chunks = [
            dialogues[i : i + chunk_size] for i in range(0, num_dialogues, chunk_size)
        ]
        logger.info(f"Number of dialogue chunks: {len(dialogue_chunks)}")
        # create temp folder
        temp_dir = self.args.cosyvoice_tmp_dir
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        # create temp dialogue shards
        dialogue_shards = []
        for i, chunk in enumerate(dialogue_chunks):
            dialogue_shard = os.path.join(temp_dir, f"input_dialogue_{i}.pkl")
            Dialogue.save_batch_to_pickle(chunk, dialogue_shard)
            dialogue_shards.append(dialogue_shard)

        # Get the path of this file
        current_path = os.path.dirname(os.path.abspath(__file__))

        # Call cosyvoice_tts_worker.py in subprocesses
        commands = []
        output_files = []
        for i, dialogue_shard in enumerate(dialogue_shards):
            output_file_name = os.path.join(temp_dir, f"output_dialogue_{i}.pkl")
            output_files.append(output_file_name)
            cmd = [
                "python",
                f"{current_path}/cosyvoice_tts_worker.py",
                "--cosyvoice_codebase",
                self.args.cosyvoice_codebase,
                "--cosyvoice_model_checkpoint",
                self.args.cosyvoice_model_checkpoint,
                "--cosyvoice_voice_bank_path",
                self.args.cosyvoice_voice_bank_path,
                "--target_sample_rate",
                str(self.target_sample_rate),
                "--input_dialogue_path",
                dialogue_shard,
                "--output_dialogue_path",
                output_file_name,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpus)
            commands.append((cmd, env))
        logger.info(f"Number of commands: {len(commands)}")
        # log the first command
        logger.info(f"First command: {' '.join(commands[0][0])}")
        processes = [subprocess.Popen(cmd, env=env) for cmd, env in commands]
        logger.info(f"Started {len(processes)} processes for TTS generation.")
        # log their pids
        for p in processes:
            logger.info(f"Started process with PID: {p.pid}")
        while processes:
            for p in processes[:]:
                retcode = p.poll()
                if retcode is not None:  # 进程已结束
                    print(f"Process {p.pid} finished with return code {retcode}")
                    processes.remove(p)
            time.sleep(5)  # 每秒检测一次进程状态
        # Read the temporary files and concatenate the audio files
        final_dialogues = []
        for output_file in output_files:
            dialogues_chunk = Dialogue.load_batch_from_pickle(output_file)
            final_dialogues.extend(dialogues_chunk)
        # Remove the temporary files
        for dialogue_shard in dialogue_shards:
            os.remove(dialogue_shard)
        for output_file in output_files:
            os.remove(output_file)
        
        # Delete the temporary directories temp_dir
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

        return final_dialogues
