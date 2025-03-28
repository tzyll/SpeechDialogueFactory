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

pipelines_fields = [
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
field2index = {field: idx for idx, (_, field) in enumerate(pipelines_fields)}


def hide_all_steps():
    outs = []
    for i in range(len(pipelines_fields)):
        outs.append(gr.update(value="", visible=False))
        outs.append(gr.update(value=None, visible=False))
    return outs


def no_change_for_steps():
    outs = []
    for i in range(len(pipelines_fields)):
        outs.append(gr.update())
        outs.append(gr.update())
    return outs


def build_step_updates(dialogue_obj, fields):
    outs = hide_all_steps()
    for field in fields:
        if field not in field2index:
            continue
        step_idx = field2index[field]
        title_pos = step_idx * 2
        content_pos = step_idx * 2 + 1
        field_name = " ".join([x.capitalize() for x in field.replace("_", " ").split()])
        outs[title_pos] = gr.update(value=f"------\n\n # {field_name}", visible=True)

        if "evaluation" in field:
            data = getattr(dialogue_obj, field).summary()
            data = {k: float(f"{v:.3f}") for k, v in data.items()}
            content_val = dict_to_markdown_yaml(data)
            outs[content_pos] = gr.update(value=content_val, visible=True)

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


def inspect_dialogue(dialogue_pkl_file, dialogue_pkl_path, current_tab):

    file_path = None
    if current_tab == "upload_pkl_tab":
        file_path = dialogue_pkl_file
    else:
        file_path = dialogue_pkl_path

    if file_path is None:
        return [
            "Please upload a valid dialogue pickle file or input a valid dialogue file path."
        ] + [gr.update() for i in range(22)]

    num_components = len(pipelines_fields) * 2
    dialogue = None
    dialogue = Dialogue.load_from_pickle(file_path)
    # try:
        
    # except Exception as e:
    #     error_message = f"Error loading dialogue pickle file: {file_path}"
    #     return [error_message] + [gr.update() for i in range(num_components)]

    if dialogue is None:
        error_message = f"Error loading dialogue pickle file: {file_path}"
        return [error_message] + [gr.update() for i in range(num_components)]

    # Check if all required files are filled in the dialogue object
    for field in pipelines_fields:
        field_name = field[1]
        if getattr(dialogue, field_name, None) is None:
            error_message = f"Error: {field_name} is not available in the dialogue."
            return [error_message] + [gr.update() for i in range(num_components)]
    
    outputs = build_step_updates(dialogue, [ field[1] for field in pipelines_fields])
    message = f"Dialogue loaded successfully from {file_path}."
    outputs = [message] + outputs
    return outputs


with gr.Blocks() as demo:
    gr.Markdown(
        "# Speech Dialogue Factory: Generate Unlimited Realistic Dialogue Data For Your Conversational LLM"
    )
    gr.Markdown("## Inspect the generated data with this page.")

    current_tab = gr.State("upload_pkl_tab")

    with gr.Tab("Upload dialogue pickle") as upload_pkl_tab:
        with gr.Column():
            gr.Markdown("### Upload the generated dialogue pickle")
            gr.Markdown(
                "You can upload the generated dialogue pickle file here. The file should be a pickle file containing one dialogue."
            )
            dialogue_pkl_file = gr.File(label="Dialogue pickle", file_types=[".pkl"])

    with gr.Tab("Input dialogue file path") as input_pkl_tab:
        with gr.Column():
            gr.Markdown("### Input the generated dialogue file path on your server")
            gr.Markdown(
                "You can input the generated dialogue file path here. The file should be a pickle file containing one dialogue."
            )
            dialogue_pkl_path = gr.Textbox(
                label="Dialogue pickle path", placeholder="Enter the file path here..."
            )

    upload_pkl_tab.select(
        lambda: "upload_pkl_tab",
        outputs=[current_tab],
    )
    input_pkl_tab.select(
        lambda: "input_pkl_tab",
        outputs=[current_tab],
    )

    inspect_btn = gr.Button(
        "Inspect Dialogue",
    )
    message_box = gr.Textbox(
        label="Message",
        interactive=False,
        placeholder="The dialogue will be displayed down below.",
    )
    gr.Markdown("--------")

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

    all_outputs = [message_box]
    for t, c in placeholders:
        all_outputs.append(t)
        all_outputs.append(c)

    inspect_btn.click(
        fn=inspect_dialogue,
        inputs=[dialogue_pkl_file, dialogue_pkl_path, current_tab],
        outputs=all_outputs,
    )
