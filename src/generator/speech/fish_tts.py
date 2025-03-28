import select
from ray import logger
import torch
import sys
import torchaudio
import re
import pandas as pd
import os
import numpy as np
import tqdm
import logging
import random
from data_classes.dialogue import Dialogue, Role
import argparse

from utils.base_classes import SDFModule


logger = logging.getLogger(__name__)

@SDFModule.set_role("generator")
class FishTTS(SDFModule):

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--fishtts_codebase",
            type=str,
            default="fish_tts",
            help="Path to the FishTTS codebase",
        )
        parser.add_argument(
            "--voice_bank_path",
            type=str,
            default="fish_tts/voice_bank",
            help="Path to the voice bank directory",
        )
        parser.add_argument(
            "--tmp_dir",
            type=str,
            default="tmp_dir",
            help="Path to the temporary directory for storing voice codes",
        )
        parser.add_argument(
            "--target_sample_rate",
            type=int,
            default=16000,
            help="Target sample rate for audio generation",
        )
        parser.add_argument(
            "--fish_tts_device",
            type=str,
            default="cuda:0",
            help="Device to run the FishTTS model on (e.g., 'cuda:0' or 'cpu')",
        )

    def __init__(self, args):
        self.args = args
        
        self.target_sample_rate = args.target_sample_rate
        

    def initialize(self):
        codebase = self.args.fishtts_codebase
        sys.path.append(self.args.fishtts_codebase)
        from tools.llama.generate import (
            load_model as load_llama_model,
            generate_long,
        )
        from tools.vqgan.inference import load_model as load_decoder_model

        # define the paths to the models
        llama_checkpoint_path = f"{codebase}/checkpoints/fish-speech-1.5/"
        decoder_checkpoint_path = f"{codebase}/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        decoder_config_name = "firefly_gan_vq"
        self.device = self.args.fish_tts_device
        compile = True
        self.llm, self.decode_one_token = load_llama_model(
            llama_checkpoint_path,
            device=self.device,
            precision=torch.bfloat16,
            compile=compile,
        )
        self.vq_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=decoder_checkpoint_path,
            device=self.device,
        )
        self.setup_speaker_retriever_commonvoice(self.args)
        self.generate_function = generate_long
        return self

    def encode_voice(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(
            audio, sr, self.vq_model.spec_transform.sample_rate
        )
        audios = audio[None].to(self.vq_model.device)
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=self.vq_model.device, dtype=torch.long
        )
        code = self.vq_model.encode(audios, audio_lengths)[0][0].cpu()
        return code

    def setup_speaker_retriever_commonvoice(self, args):
        voice_bank_path = args.voice_bank_path
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)

        tsv_file = f"{voice_bank_path}/validated.tsv"
        clips = f"{voice_bank_path}/clips"
        df = pd.read_csv(tsv_file, sep="\t")
        df = pd.read_csv(tsv_file, sep="\t")
        df = df.dropna(subset=["path", "sentence", "age", "gender"])[
            ["client_id", "path", "sentence", "age", "gender"]
        ]
        df["path"] = df["path"].apply(lambda x: f"{clips}/{x}")
        logger.info(
            f"Statistics of the voice bank: Ages:{df['age'].unique}, Genders:{df['gender'].unique}, Number of voices:{len(df['client_id'].unique())}"
        )

        if not os.path.exists(f"{tmp_path}/voice_codes.pt"):
            print("Encoding voices")
            voice_codes = []
            for p in tqdm.tqdm(df["path"].tolist()):
                voice_codes.append(self.encode_voice(p))

            # Save voice codes to tmp directory
            voice_codes_path = f"{tmp_path}/voice_codes.pt"
            torch.save(voice_codes, voice_codes_path)
            print(f"Voice codes saved to {voice_codes_path}")
        else:
            print("Loading voice codes")
            voice_codes = torch.load(f"{tmp_path}/voice_codes.pt")
            print("Voice codes loaded")
        df["codes"] = voice_codes

        possible_ages_r = {
            10: "teens",
            20: "twenties",
            30: "thirties",
            40: "fourties",
            50: "fifties",
            60: "sixties",
            70: "seventies",
            80: "eighties",
            90: "nineties",
        }
        possible_ages = {v: k for k, v in possible_ages_r.items()}

        df["age"] = df["age"].apply(lambda x: possible_ages[x])
        self.voice_bank = df

    def retrieve_speaker(self, role: Role):
        voice_df = self.voice_bank.copy()
        voice_df["age_diff"] = voice_df["age"].apply(lambda x: abs(x - role.age))
        voice_df = voice_df.sort_values("age_diff")

        # select gender
        voice_df["matched_gender"] = voice_df["gender"].apply(
            lambda x: x.startswith(role.gender)
        )
        candidates = voice_df[voice_df["matched_gender"]]

        if len(candidates) == 0:
            logger.info(f"No appropriate voices found")
            rand_idx = random.choice(list(range(len(voice_df))))
            selected_voice_info = voice_df.iloc[rand_idx]

        else:
            rand_idx = random.choice(list(range(len(candidates)))[:5])
            selected_voice_info = candidates.iloc[rand_idx]
        logger.info(
            f"Selected voice: {selected_voice_info['client_id']} {selected_voice_info['age']} {selected_voice_info['gender']}"
        )

        voice_path = selected_voice_info["path"]
        voice_speech = selected_voice_info["codes"]
        voice_text = selected_voice_info["sentence"]
        return {
            "voice_id": selected_voice_info["client_id"],
            "voice_path": voice_path,
            "voice_text": voice_text,
            "voice_speaker_age": selected_voice_info["age"],
            "voice_speaker_gender": selected_voice_info["gender"],
            "speaker_name": role.name,
            "speaker_gender": role.gender,
            "speaker_age": role.age,
            "speaker_nationality": role.nationality,
            "speaker_occupation": role.occupation,
            "speaker_personality_traits": ",".join(role.personality_traits),
        }, voice_speech

    @torch.no_grad()
    def generate_utterance(self, text, prompt_text, prompt_tokens):

        num_samples = 1
        max_new_tokens = 512
        top_p = 0.7
        repetition_penalty = 1.7
        temperature = 0.7
        compile = True
        iterative_prompt = True
        chunk_length = 100
        device = self.device

        generator = self.generate_function(
            model=self.llm,
            device=device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=compile,
            iterative_prompt=iterative_prompt,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )

        idx = 0
        codes = [[]]

        for response in generator:
            if response.action == "sample":
                codes[-1].append(response.codes)
            elif response.action == "next":
                codes.append([])
                idx += 1
            else:
                print("error")
        indices = torch.cat(codes[0], 1)
        feature_lengths = torch.tensor([indices.shape[1]], device=device)
        generated_audios, _ = self.vq_model.decode(
            indices=indices[None], feature_lengths=feature_lengths
        )
        generated_audio = generated_audios[0, 0].float().cpu().view(-1)

        # Resample to target sample rate
        generated_audio = torchaudio.functional.resample(
            generated_audio.unsqueeze(0),
            self.vq_model.spec_transform.sample_rate,
            self.target_sample_rate,
        ).squeeze(0)
        return generated_audio

    def synthesize_one_dialogue(self, dialogue: Dialogue):
        metadata = dialogue.metadata
        conversation = dialogue.conversation
        role_1 = metadata.role_1
        role_2 = metadata.role_2
        role_1_voice_profile, role1_voice_speech = self.retrieve_speaker(role_1)
        role_2_voice_profile, role2_voice_speech = self.retrieve_speaker(role_2)

        voice_profiles = {
            "role_1": (role_1_voice_profile, role1_voice_speech),
            "role_2": (role_2_voice_profile, role2_voice_speech),
        }

        PAUSE_AFTER_MULTIPLIER = {
            "interrupted": 0.0,
            "short": 0.5,
            "medium": 1.0,
            "long": 1.5,
        }
        PAUSE_BASE = 150 / 1000

        speech_rates = {"slow": 1.0, "medium": 1.05, "fast": 1.1}

        synthesized_utterances = []

        for idx, utterance in enumerate(conversation.utterances):
            speaker_id = utterance.speaker_id
            speaker_name = utterance.speaker_name
            text = utterance.text
            emotion = utterance.emotion
            speech_rate = utterance.speech_rate
            pause_after = utterance.pause_after
            tts_prompt = utterance.tts_prompt

            voice_profile, voice_speech = voice_profiles[speaker_id]
            voice_text = voice_profile["voice_text"]
            characteristics = voice_profile["speaker_personality_traits"]
            age = voice_profile["speaker_age"]
            gender = voice_profile["speaker_gender"]
            occupation = voice_profile["speaker_occupation"]
            nationality = voice_profile["speaker_nionality"]

            # Generate the audio, tts_prompt is not usable for fish tts
            waveform = self.generate_utterance(text, voice_text, voice_speech)
            # if concat:
            #     # Add pauses
            #     pause_after_ratio = PAUSE_AFTER_MULTIPLIER.get(pause_after, 1.0)
            #     pause_duration = PAUSE_BASE * pause_after_ratio
            #     pause = torch.zeros(int(pause_duration * self.target_sample_rate))
            #     waveform = torch.cat([waveform, pause], dim=-1)
            synthesized_utterances.append(waveform)
        synthesized_utterances = [
            wave.cpu().view(-1).numpy() for wave in synthesized_utterances
        ]
        return {
            "waveforms": synthesized_utterances,
            "sample_rate": self.target_sample_rate,
            "voice_profiles": {
                "role_1": role_1_voice_profile,
                "role_2": role_2_voice_profile,
            },
        }

    def generate(self, dialogues: list[Dialogue], gen_params={}):
        """
        Synthesize a list of dialogues using the CosyVoice TTS model.
        """
        all_synthesized_utterances = []
        for dialogue in dialogues:
            synthesized_utterance = self.synthesize_one_dialogue(dialogue)
            all_synthesized_utterances.append(synthesized_utterance)
        return all_synthesized_utterances