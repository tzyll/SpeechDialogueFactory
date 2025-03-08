import os
import sys
import torchaudio
import re
import pandas as pd
import logging
import random
from data.dialogue import Dialogue, Role
import torch

logger = logging.getLogger(__name__)

all_available_markers = [
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
]


def extract_tags(text):
    pattern = r"\[.*?\]|<.*?>"
    return re.findall(pattern, text)


class CosyVoiceTTS:
    def __init__(self, args):
        args.cosyvoice_codebase = args.cosyvoice_codebase
        sys.path.append(args.cosyvoice_codebase)
        sys.path.append(
            os.path.join(args.cosyvoice_codebase, "third_party/Matcha-TTS/")
        )
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        self.cosyvoice_model = CosyVoice2(
            args.cosyvoice_model_checkpoint,
            load_jit=True,
            load_onnx=False,
            load_trt=False,
        )
        self.load_wav = load_wav
        self.setup_speaker_retriever_commonvoice(args)
        self.target_sample_rate = args.target_sample_rate

    def setup_speaker_retriever_commonvoice(self, args):
        voice_bank_path = args.voice_bank_path
        tsv_file = f"{voice_bank_path}/validated.tsv"
        clips = f"{voice_bank_path}/clips"
        df = pd.read_csv(tsv_file, sep="\t")
        df = df.dropna(subset=["path", "sentence", "age", "gender"])[
            ["client_id", "path", "sentence", "age", "gender"]
        ]
        df["path"] = df["path"].apply(lambda x: f"{clips}/{x}")
        logger.info(
            f"Statistics of the voice bank: Ages:{df['age'].unique}, Genders:{df['gender'].unique}, Number of voices:{len(df['client_id'].unique())}"
        )

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
        voice_speech = self.load_wav(voice_path, 16000)
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
    def synthesize(self, dialogue: Dialogue, concat=False):
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

            speech_markers = extract_tags(text)

            # filter markers that do not belongs to all_available_markers
            invalid_speech_markers = [
                marker
                for marker in speech_markers
                if marker not in all_available_markers
            ]
            # remove invalid markers from text
            for marker in invalid_speech_markers:
                text = text.replace(marker, "")

            # Remove actions from text where actions are often enclosed in ()
            text = re.sub(r"\(.*?\)", "", text)

            profile_template = "The speaker is a {age} year old {gender} {nationality} speaker, who is a {occupation} and has the personality traits: {characteristics}.\n{tts_prompt}"
            control_instruct = profile_template.format(
                age=age,
                gender=gender,
                nationality=nationality,
                occupation=occupation,
                characteristics=characteristics,
                tts_prompt=tts_prompt,
            )
            waveform = next(
                self.cosyvoice_model.inference_instruct2(
                    text,
                    control_instruct,
                    voice_speech,
                    stream=False,
                    speed=speech_rates[speech_rate],
                )
            )["tts_speech"].cpu().view(-1)
            # Resample to target sample rate
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=self.cosyvoice.sample_rate,
                new_freq=self.target_sample_rate,
            ).squeeze(0)
            if concat:
                # Add pauses
                pause_after_ratio = PAUSE_AFTER_MULTIPLIER.get(pause_after, 1.0)
                pause_duration = PAUSE_BASE * pause_after_ratio
                pause = torch.zeros(int(pause_duration * self.target_sample_rate))
                waveform = torch.cat([waveform, pause], dim=-1)
            synthesized_utterances.append(waveform)
        if concat:
            synthesized_utterances = torch.cat(synthesized_utterances, dim=-1).cpu().view(-1).numpy()
        else:
            synthesized_utterances = [wave.cpu().view(-1).numpy() for wave in synthesized_utterances]
        return {
            "synthesized_utterances": synthesized_utterances,
            "voice_profiles": {
                "role_1": role_1_voice_profile,
                "role_2": role_2_voice_profile,
            },
        }

