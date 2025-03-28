import gradio as gr
import logging
from speech_dialogue_factory import SpeechDialogueFactory, create_sdf
from data_classes.dialogue import Dialogue
import queue
import threading
import time
import argparse
import pandas as pd
from utils.misc import dict_to_markdown_yaml
import numpy as np

logger = logging.getLogger(__name__)


def add_arguments(parser):
    """
    Add arguments for the factory
    """
    parser.add_argument(
        "--sdf_config",
        type=str,
        default="configs/sdf_config.json",
        help="Path to the SDF config file.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Run in dryrun mode.",
    )
    parser.add_argument(
        "--dryrun_sample_path",
        type=str,
        default="./output/dialogue_0.pkl",
        help="Path to the dryrun sample file.",
    )


class SDFApp:
    def __init__(self, args):
        self.sdf_config = args.sdf_config
        self.dryrun = args.dryrun
        self.dryrun_sample_path = args.dryrun_sample_path
        self.sdf = None
        self.result = None
        self.dryrun_sample = None
        if not self.dryrun:
            self.sdf = create_sdf(self.sdf_config)
        else:
            self.dryrun_sample = Dialogue.load_from_pickle(self.dryrun_sample_path)
            logger.info(f"Load dryrun sample from {self.dryrun_sample_path} for UI.")
        self.is_running = False
        self.queue = queue.Queue()

        self.pipelines_fields = [
            ("ScenarioGenerator", "scenario"),
            ("MetadataGenerator", "metadata"),
            ("ScriptGenerator", "script"),
            ("DialogueGenerator", "conversation"),
            ("ConsistencyEvaluator", "consistency_evaluation"),
            ("CoherenceEvaluator", "coherence_evaluation"),
            ("NaturalnessEvaluator", "naturalness_evaluation"),
            ("TTS", "dialogue_audio"),
            ("IntelligibilityEvaluator", "intelligibility_evaluation"),
            ("SpeakerConsistencyEvaluator", "speaker_consistency_evaluation"),
            ("SpeechQualityEvaluator", "speech_quality_evaluation"),
        ]
        self.field2index = {
            field: idx for idx, (_, field) in enumerate(self.pipelines_fields)
        }

    def generate_one_dialogue(self, custom_prompt, language):
        def run_generation():
            callback = lambda msg: self.queue.put(msg)
            dialogue, final_message = self.sdf.generate_sample_dialogue(
                num_dialogues=1,
                dialogue_languages=[language],
                custom_prompts=[custom_prompt],
                process_callback=callback,
            )
            self.result = dialogue
            self.queue.put(final_message)
            self.is_running = False

        def run_generation_dryrun():
            dialogue = self.dryrun_sample
            finished_fields = []
            for i, (name, field) in enumerate(self.pipelines_fields):
                time.sleep(1)
                finished_fields.append(field)
                msg = {
                    "status": "generating",
                    "current_step": i + 1,
                    "total_steps": len(self.pipelines_fields),
                    "message": f"**Status: Processing with {name}...**",
                    "dialogues": dialogue,
                    "finished_fields": finished_fields,
                    "saved_dialogues": None,
                }
                self.queue.put(msg)
            self.queue.put(
                {
                    "status": "complete",
                    "current_step": len(self.pipelines_fields),
                    "total_steps": len(self.pipelines_fields),
                    "message": "**Status: Generation complete!**",
                    "dialogues": dialogue,
                    "finished_fields": finished_fields,
                    "saved_dialogues": self.dryrun_sample_path,
                }
            )
            self.is_running = False

        run_function = run_generation if not self.dryrun else run_generation_dryrun
        self.is_running = True
        self.result = None
        self.queue = queue.Queue()
        thread = threading.Thread(target=run_function)
        thread.daemon = True
        thread.start()

    def get_updates(self):
        """Retrieve all messages from the queue"""
        updates = []
        try:
            while True:
                updates.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return updates

    def reset(self):
        self.is_running = False
        self.result = None
        self.queue = queue.Queue()

    def hide_all_steps(self):
        outs = []
        for i in range(len(self.pipelines_fields)):
            outs.append(gr.update(value="", visible=False))
            outs.append(gr.update(value=None, visible=False))
        return outs

    def no_change_for_steps(self):
        outs = []
        for i in range(len(self.pipelines_fields)):
            outs.append(gr.update())
            outs.append(gr.update())
        return outs

    def build_step_updates(self, dialogue_obj, finished_fields):
        outs = self.hide_all_steps()
        for field in finished_fields:
            if field not in self.field2index:
                continue
            step_idx = self.field2index[field]
            title_pos = step_idx * 2
            content_pos = step_idx * 2 + 1
            field_name = " ".join(
                [x.capitalize() for x in field.replace("_", " ").split()]
            )
            outs[title_pos] = gr.update(
                value=f"------\n\n # {field_name}", visible=True
            )

            if "evaluation" in field:
                data = getattr(dialogue_obj, field).summary()
                data = {k: float(f"{v:.3f}") for k, v in data.items()}
                content_val = dict_to_markdown_yaml(data)
                outs[content_pos] = gr.update(value=content_val, visible=True)
                # df = pd.DataFrame(list(data.items()), columns=["key", "value"])
                # outs[content_pos] = gr.update(
                #     value=df, x="key", y="value", visible=True
                # )

            elif field == "dialogue_audio":
                audio_dict = dialogue_obj.dialogue_audio
                waveforms = audio_dict["waveforms"]
                sr = audio_dict.get("sample_rate", 16000)
                audio_data = np.concatenate(waveforms, axis=0)
                outs[content_pos] = gr.update(value=(sr, audio_data), visible=True)
            else:
                content_val = getattr(dialogue_obj, field)
                content_val = (
                    content_val
                    if isinstance(content_val, str)
                    else dict_to_markdown_yaml(content_val.to_dict())
                )
                outs[content_pos] = gr.update(value=content_val, visible=True)
        return outs

    def start_generation(self, prompt, language):
        self.generate_one_dialogue(prompt, language)
        return [
            0.0,
            "**Status: Start to generate dialogue...**",
            gr.update(
                interactive=False,
                value="Please wait patiently, the generation may take a few minutes.",
            ),
            gr.update(
                active=True,
            ),
            gr.update(interactive=False),
        ]

    def get_progress_updates(self):
        updates = self.get_updates()
        if not updates:
            return [gr.update() for i in range(5)] + self.no_change_for_steps()

        progress_val = 0
        status_msg = ""
        dialogue_obj = None
        finished_fields = []
        button_active = False
        button_text = "Please wait patiently, the generation may take a few minutes."
        timer_active = True
        download_path = None

        for data in updates:
            update_type = data["status"]
            progress_val = (data["current_step"] / data["total_steps"]) * 100
            status_msg = data["message"]
            dialogue_obj = data["dialogues"]
            finished_fields = data["finished_fields"]
            if update_type == "complete":
                self.is_running = False
                self.result = None
                self.reset()
                download_path = data["saved_dialogues"]
                button_active = True
                button_text = "Generate"
                timer_active = False
                break
        step_updates = self.build_step_updates(dialogue_obj, finished_fields)

        return [
            progress_val,
            status_msg,
            gr.update(interactive=button_active, value=button_text),
            gr.update(active=timer_active),
            gr.update(interactive=button_active, value=download_path),
        ] + step_updates


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

# sdf_app = SDFApp(args)
sdf_app = SDFApp(
    argparse.Namespace(
        sdf_config="configs/sdf_config_app_oai.json",
        dryrun=True,
        dryrun_sample_path="./output/dialogue_0.pkl",
    )
)

with gr.Blocks() as demo:
    gr.Markdown(
        "# Speech Dialogue Factory: Generate Unlimited Realistic Dialogue Data For Your Conversational LLM"
    )
    gr.Markdown("## Generate a sample dialogue with a custom prompt and language.")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Custom Prompt",
                placeholder="e.g. 'Generate a dialogue between a father and a son about a famous video game named \"War Thunder\".'",
                lines=1,
            )
            language = gr.Dropdown(
                label="Language",
                choices=["English", "Chinese"],
                value="English",
            )
            generate_button = gr.Button("Generate", interactive=True)
            download_button = gr.DownloadButton(
                label="Download Dialogue", interactive=False
            )

    status_text = gr.Markdown(value="**Status: Waiting for the input...**")
    progress_bar = gr.Slider(
        label="Progress",
        minimum=0,
        maximum=100,
        value=0.0,
        interactive=False,
    )

    timer = gr.Timer(value=1.0, active=False)

    placeholders = []
    with gr.Blocks():
        pipeline_to_component = {
            "scenario": gr.Markdown,
            "metadata": gr.Markdown,
            "script": gr.Markdown,
            "conversation": gr.Markdown,
            "consistency_evaluation": gr.Markdown,
            "coherence_evaluation": gr.Markdown,
            "naturalness_evaluation": gr.Markdown,
            "dialogue_audio": gr.Audio,
            "intelligibility_evaluation": gr.Markdown,
            "speaker_consistency_evaluation": gr.Markdown,
            "speech_quality_evaluation": gr.Markdown,
        }
        for pipeline_name, component in pipeline_to_component.items():
            title = gr.Markdown(visible=False)
            content = component(visible=False)
            placeholders.append((title, content))

    all_outputs = [progress_bar, status_text, generate_button, timer, download_button]
    for t, c in placeholders:
        all_outputs.append(t)
        all_outputs.append(c)

    generate_button.click(
        fn=sdf_app.start_generation,
        inputs=[prompt_input, language],
        outputs=all_outputs[:5],
    )

    timer.tick(
        fn=sdf_app.get_progress_updates,
        outputs=all_outputs,
    )
# Generate a dialouge between a father and a son about a famous video game named "War Thunder".
