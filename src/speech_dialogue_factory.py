from gguf import OrderedDict
from more_itertools import last
from sympy import N, Order
import tqdm
from data_classes.dialogue import Dialogue
from evaluator.content.content_quality_filter import ContentQualityFilter
from evaluator.speech.speech_quality_filter import SpeechQualityFilter
from generator.content.dialogue_generator import DialogueGenerator
from generator.content.metadata_generator import MetadataGenerator
from generator.content.scenario_generator import ScenarioGenerator
from generator.content.script_generator import ScriptGenerator
from generator.speech.cosyvoice_tts import CosyVoiceTTS
from generator.speech.fish_tts import FishTTS

from evaluator.content.coherence_evaluator import CoherenceEvaluator
from evaluator.content.consistency_evaluator import ConsistencyEvaluator
from evaluator.content.naturalness_evaluator import NaturalnessEvaluator
from evaluator.speech.intelligibility_evaluator import IntelligibilityEvaluator
from evaluator.speech.speaker_consistency_evaluator import SpeakerConsistencyEvaluator
from evaluator.speech.speech_quality_evaluator import SpeechQualityEvaluator
from utils.llm import LLM
import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


class SpeechDialogueFactory:
    """Factory class to create and manage dialogue and speech generation components."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser = None):
        """Add command-line arguments for the factory."""
        if parser is None:
            parser = argparse.ArgumentParser()
        # Add arguments for the LLM
        parser.add_argument(
            "--sdf_config",
            type=str,
            default="./configs/sdf_config.json",
            help="Path to the configuration file",
        )
        parser.add_argument(
            "--tts_in_use",
            type=str,
            default="cosyvoice",
            choices=["cosyvoice", "fish"],
            help="TTS model to use for speech synthesis",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./output",
            help="Directory to save the generated dialogues and evaluations",
        )
        parser.add_argument(
            "--input_prompt_file",
            type=str,
            default=None,
            help="Path to the input prompt file for batch generation",
        )
        parser.add_argument(
            "--num_dialogues_per_prompt",
            type=int,
            default=1,
            help="Number of dialogues to generate per prompt",
        )
        parser.add_argument(
            "--dialogue_language",
            type=str,
            default="English",
            choices=["English", "Chinese"],
            help="Language for the generated dialogues",
        )
        parser.add_argument(
            "--lazy_load",
            action="store_true",
            help="Lazy load the models to save memory",
        )

    def __init__(self, args):
        if args.sdf_config is not None:
            print("Loading SDF config from:", args.sdf_config)
            with open(args.sdf_config, "r") as f:
                config = json.load(f)
            sdf_args = config.get("sdf_args", {})
            module_configs = config.get("module_args", {})
            module_args = {}
            for key, value in module_configs.items():
                for k, v in value.items():
                    module_args[k] = v
            # Flatten config groups into a single namespace
            sdf_args = argparse.Namespace(**sdf_args)
            module_args = argparse.Namespace(**module_args)
            # Override args with sdf_args
            for key, value in vars(sdf_args).items():
                setattr(args, key, value)

        else:
            # Parse args for all components
            parser = argparse.ArgumentParser()
            DialogueGenerator.add_arguments(parser)
            MetadataGenerator.add_arguments(parser)
            ScenarioGenerator.add_arguments(parser)
            ScriptGenerator.add_arguments(parser)
            CosyVoiceTTS.add_arguments(parser)
            FishTTS.add_arguments(parser)
            CoherenceEvaluator.add_arguments(parser)
            ConsistencyEvaluator.add_arguments(parser)
            NaturalnessEvaluator.add_arguments(parser)
            ContentQualityFilter.add_arguments(parser)
            IntelligibilityEvaluator.add_arguments(parser)
            SpeakerConsistencyEvaluator.add_arguments(parser)
            SpeechQualityEvaluator.add_arguments(parser)
            SpeechQualityFilter.add_arguments(parser)
            LLM.add_arguments(parser)
            module_args = parser.parse_args()

        self.TTS_module = {
            "cosyvoice": CosyVoiceTTS,
            "fish": FishTTS,
        }.get(args.tts_in_use, CosyVoiceTTS)

        self.lazy_load = args.lazy_load

        self.llm = LLM(module_args).initialize()
        self.scenario_generator = ScenarioGenerator(module_args, self.llm).initialize()
        self.metadata_generator = MetadataGenerator(module_args, self.llm).initialize()
        self.script_generator = ScriptGenerator(module_args, self.llm).initialize()
        self.dialogue_generator = DialogueGenerator(module_args, self.llm).initialize()
        self.coherence_evaluator = CoherenceEvaluator(
            module_args, self.llm
        ).initialize()
        self.consistency_evaluator = ConsistencyEvaluator(
            module_args, self.llm
        ).initialize()
        self.naturalness_evaluator = NaturalnessEvaluator(
            module_args, self.llm
        ).initialize()
        self.content_quality_filter = ContentQualityFilter(
            module_args, self.llm
        ).initialize()
        self.speech_quality_filter = SpeechQualityFilter(
            module_args, self.llm
        ).initialize()
        self.tts = self.TTS_module(module_args)
        self.intelligibility_evaluator = IntelligibilityEvaluator(module_args)
        self.speaker_consistency_evaluator = SpeakerConsistencyEvaluator(module_args)
        self.speech_quality_evaluator = SpeechQualityEvaluator(module_args)
        if not self.lazy_load:
            self.tts.initialize()
            self.intelligibility_evaluator.initialize()
            self.speaker_consistency_evaluator.initialize()
            self.speech_quality_evaluator.initialize()

        self.sdf_args = args
        self.module_args = module_args

        # Set the output directory
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_sample_dialogue(
        self,
        num_dialogues=1,
        dialogue_languages=None,
        custom_prompts=None,
        process_callback=None,
    ):

        pipelines = [
            ("ScenarioGenerator", self.scenario_generator, "scenario", "content"),
            ("MetadataGenerator", self.metadata_generator, "metadata", "content"),
            ("ScriptGenerator", self.script_generator, "script", "content"),
            ("DialogueGenerator", self.dialogue_generator, "conversation", "content"),
            (
                "ConsistencyEvaluator",
                self.consistency_evaluator,
                "consistency_evaluation",
                "content",
            ),
            (
                "CoherenceEvaluator",
                self.coherence_evaluator,
                "coherence_evaluation",
                "content",
            ),
            (
                "NaturalnessEvaluator",
                self.naturalness_evaluator,
                "naturalness_evaluation",
                "content",
            ),
            ("TTS", self.tts, "dialogue_audio", "speech"),
            (
                "IntelligibilityEvaluator",
                self.intelligibility_evaluator,
                "intelligibility_evaluation",
                "speech",
            ),
            (
                "SpeakerConsistencyEvaluator",
                self.speaker_consistency_evaluator,
                "speaker_consistency_evaluation",
                "speech",
            ),
            (
                "SpeechQualityEvaluator",
                self.speech_quality_evaluator,
                "speech_quality_evaluation",
                "speech",
            ),
        ]

        # Generate dialogues
        input_args = {
            "num_dialogues": num_dialogues,
            "dialogue_languages": dialogue_languages,
            "custom_prompts": custom_prompts,
        }
        dialogues = []
        finished_fields = []
        last_process_type = "content"
        for i, (name, module, field, process_type) in enumerate(pipelines):
            logger.info(f"Working with {name}...")

            if self.lazy_load:
                # Unload the llm to release GPU when turning from content to speech
                if last_process_type != process_type:
                    self.llm.unload()
                module.initialize()
                last_process_type = process_type

            if module.role == "generator":
                dialogues = module.generate(**input_args)
            elif module.role == "evaluator":
                dialogues = module.evaluate(**input_args)
            input_args = {
                "dialogues": dialogues,
            }
            finished_fields.append(field)
            message = (
                f"**Status: Processing with {pipelines[i + 1][0]}...**"
                if i + 1 < len(pipelines)
                else "**Status: Packing up...**"
            )
            if process_callback:
                process_callback(
                    {
                        "status": "generating",
                        "current_step": i + 1,
                        "total_steps": len(pipelines),
                        "message": message,
                        "dialogues": dialogues[0],
                        "finished_fields": finished_fields,
                        "saved_dialogues": None,
                    }
                )
            logger.info(f"Finished {name}...")
        # Set a dialogue ID for each dialogue
        for i, dialogue in enumerate(dialogues):
            dialogue.dialogue_id = str(i)

        # Save dialogues to the output directory
        saved_dialogues = []
        for i, dialogue in enumerate(dialogues):
            dialogue_path = os.path.join(self.output_dir, f"dialogue_{i}.pkl")
            dialogue.save_to_pickle(dialogue_path)
            saved_dialogues.append(dialogue_path)
        final_message = {
            "status": "complete",
            "current_step": len(pipelines),
            "total_steps": len(pipelines),
            "message": "**Status: Generation complete!**",
            "dialogues": dialogues[0],
            "finished_fields": finished_fields,
            "saved_dialogues": saved_dialogues[0],
        }

        return dialogues[0], final_message

    def generate_batched_dialogues(
        self, input_prompt_file, language, num_dialogues_per_prompt=1
    ):
        """
        Generate dialogues in batches.
        Args:
            input_prompt_file (str): Path to the input prompt file.
            language (str): Language for the dialogues.
        """
        # Read the prompts from the input file
        task_id = input_prompt_file.split("/")[-1].split(".")[0]
        with open(input_prompt_file, "r", encoding="utf-8") as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts if p.strip()]

        # Multiply the number of dialogues per prompt
        num_dialogues_per_prompt = int(num_dialogues_per_prompt)
        prompts = [[prompt] * num_dialogues_per_prompt for prompt in prompts]
        prompts = [prompt for sublist in prompts for prompt in sublist]
        num_dialogues = len(prompts)
        dialogue_languages = [language] * num_dialogues
        pipelines = [
            ("ScenarioGenerator", self.scenario_generator, "content"),
            ("MetadataGenerator", self.metadata_generator, "content"),
            ("ScriptGenerator", self.script_generator, "content"),
            ("DialogueGenerator", self.dialogue_generator, "content"),
            (
                "ConsistencyEvaluator",
                self.consistency_evaluator,
                "content",
            ),
            ("CoherenceEvaluator", self.coherence_evaluator, "content"),
            (
                "NaturalnessEvaluator",
                self.naturalness_evaluator,
                "content",
            ),
            (
                "ContentQualityFilter",
                self.content_quality_filter,
                "content",
            ),
            ("TTS", self.tts, "speech"),
            (
                "IntelligibilityEvaluator",
                self.intelligibility_evaluator,
                "speech",
            ),
            (
                "SpeakerConsistencyEvaluator",
                self.speaker_consistency_evaluator,
                "speech",
            ),
            ("SpeechQualityEvaluator", self.speech_quality_evaluator, "speech"),
            ("SpeechQualityFilter", self.speech_quality_filter, "speech"),
        ]
        # Make dir for the output task
        task_dir = os.path.join(self.output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        intermediate_dir = os.path.join(task_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        # Generate dialogues
        input_args = {
            "num_dialogues": num_dialogues,
            "dialogue_languages": dialogue_languages,
            "custom_prompts": prompts,
        }
        dialogues = []
        last_process_type = "content"

        start_step = 0

        # Resume the generation from the last step
        if os.path.exists(intermediate_dir):
            intermediate_files = os.listdir(intermediate_dir)
            if len(intermediate_files)>0:
                # Find the last step according to the file name
                last_step_file = sorted(intermediate_files, key=lambda x: int(x.split("_")[2]))[-1]
                last_step = int(last_step_file.split("_")[2])
                dialogues = Dialogue.load_batch_from_pickle(
                    os.path.join(intermediate_dir, last_step_file)
                )
                start_step = last_step + 1
                logger.info(f"Resuming from step {last_step}...")
                input_args = {
                    "dialogues": dialogues,
                }
                last_process_type = pipelines[start_step-1][-1]


        for i in range(start_step, len(pipelines)):
            name, module, process_type = pipelines[i]
            logger.info(f"Working with {name}...")
            if self.lazy_load:
                # Unload the llm to release GPU when turning from content to speech
                if last_process_type != process_type:
                    self.llm.unload()
                module.initialize()
                last_process_type = process_type
            if module.role == "generator":
                dialogues = module.generate(**input_args)
            elif module.role == "evaluator":
                dialogues = module.evaluate(**input_args)
            # Save intermediate dialogues to the output directory with a Batched version
            Dialogue.save_batch_to_pickle(
                dialogues,
                os.path.join(intermediate_dir, f"dialogues_step_{i}_{name}.pkl"),
            )
            input_args = {
                "dialogues": dialogues,
            }
            logger.info(f"Finished {name}...")

        # Set a dialogue ID for each dialogue
        for i, dialogue in enumerate(dialogues):
            dialogue.dialogue_id = str(i)

        final_dir = os.path.join(task_dir, "generated_dialogues")
        os.makedirs(final_dir, exist_ok=True)
        # Save dialogues to the output directory
        for i, dialogue in enumerate(dialogues):
            dialogue_path = os.path.join(final_dir, f"dialogue_{i}.pkl")
            dialogue.save_to_pickle(dialogue_path)
        # Save the metadata to a JSON file
        metadata_path = os.path.join(task_dir, "task_report.json")
        task_report = {
            "task_id": task_id,
            "num_planned_dialogues": num_dialogues,
            "num_generated_dialogues": len(dialogues),
            "output_dir": task_dir,
        }
        with open(metadata_path, "w") as f:
            json.dump(task_report, f, indent=4)

        return task_report


def create_sdf(sdf_config_path):
    factory = SpeechDialogueFactory(argparse.Namespace(sdf_config=sdf_config_path))
    return factory


def batch_cli_main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Speech Dialogue Factory CLI")
    # Add arguments for the factory
    SpeechDialogueFactory.add_arguments(parser)
    # Parse the arguments
    args = parser.parse_args()
    # Create an instance of the factory
    factory = SpeechDialogueFactory(args)
    # Generate dialogues in batches
    task_report = factory.generate_batched_dialogues(
        args.input_prompt_file,
        args.dialogue_language,
        args.num_dialogues_per_prompt,
    )
    # Print the task report
    print("Task Report:")
    print(json.dumps(task_report, indent=4))


if __name__ == "__main__":
    # Run the CLI main function
    batch_cli_main()
