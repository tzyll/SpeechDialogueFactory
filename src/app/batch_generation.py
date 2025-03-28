import gradio as gr
from data_classes.dialogue import Dialogue
import queue
import threading
import time
import numpy as np
import os
import uuid

def generate_command_line(
    dialogue_language,
    custom_prompt,
    custom_prompt_file,
    num_dialogues_per_prompt,
    current_tab,
):

    prompt_file_path = None
    num_prompts = 0
    num_total_dialogues = 0
    prompt_file_id = None
    if current_tab == "input_prompt_tab":
        if custom_prompt is None or custom_prompt.strip() == "":
            return "**Please enter a valid custom prompt.**"
        prompts = custom_prompt.split("\n")
        prompts = [p.strip() for p in prompts if p.strip()]
        # Save prompts to a file
        os.makedirs("./tmp", exist_ok=True)
        prompt_file_id = str(uuid.uuid4())
        prompt_file_path = f"./tmp/{prompt_file_id}.txt"
        with open(prompt_file_path, "w") as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        num_prompts = len(prompts)
    elif current_tab == "upload_prompt_tab":
        if custom_prompt_file is None or len(custom_prompt_file) == 0:
            return "**Please upload a valid custom prompt file.**"
        prompt_file_path = custom_prompt_file
        # Check the content of the file
        with open(prompt_file_path, "r") as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts if p.strip()]
            if len(prompts) < 1:
                return "**The uploaded file is empty or contains no valid prompts.**"
            if len(prompts) > 2000:
                return "**The uploaded file contains too many prompts. Please limit to 2000.**"
            # Save to ./tmp with a uuid as name
            prompt_file_id = str(uuid.uuid4())
            prompt_file_path = f"./tmp/{prompt_file_id}.txt"
            with open(prompt_file_path, "w") as f:
                for prompt in prompts:
                    f.write(f"{prompt}\n")
        num_prompts = len(prompts)
    else:
        return "**Please select a valid prompt tab.**"
    num_total_dialogues = num_prompts * num_dialogues_per_prompt
    message = """
    ### Command Line
    ```bash
    PYTHONPATH=./src/ python src/speech_dialogue_factory.py \ 
    --sdf_config ./configs/sdf_config.json \ 
    --input_prompt_file {prompt_file_path} \ 
    --num_dialogues_per_prompt {num_dialogues_per_prompt} \ 
    --dialogue_language {dialogue_language} \ 
    --output_dir ./output/
    ```

    ### Input Prompt Information
    - **Number of Validated Input Prompts**: {num_prompts}
    - **Planned Total Number of Dialogues**: {num_total_dialogues}
    - **Task ID**: {prompt_file_id}

    ### Explanation
    Please run the above command line at the root path of the cloned SDF repository.
    The generated dialogue data will be saved in the `./output/<task_id>` directory by default, where the `<task_id>` is a unique identifier for the task, same as the name of the prompt file.
    You can adjust the parameters in the command line as needed.
    The generation will take some time depending on the number of dialogues and the hardware you are using, please be patient.
    """

    # Generate command line
    command_line = message.format(
        prompt_file_path=prompt_file_path,
        num_dialogues_per_prompt=num_dialogues_per_prompt,
        dialogue_language=dialogue_language,
        num_prompts=num_prompts,
        num_total_dialogues=num_total_dialogues,
        prompt_file_id=prompt_file_id,
    )
    # Return the command line
    return command_line


with gr.Blocks() as demo:
    gr.Markdown(
        "# Speech Dialogue Factory: Generate Unlimited Realistic Dialogue Data For Your Conversational LLM"
    )
    gr.Markdown("## Generate large-scale dialogue data with a few clicks")

    gr.Markdown(
        "Since the generation of dialogue data is a time-consuming process, we have provided a customized interface to help you adjust the parameters for batch generation. You will get a generated command line to run the batch generation on your server with SDF installed on it."
    )
    gr.Markdown("## Batch Generation Parameters")

    gr.Markdown("### Dialogue Language")
    gr.Markdown(
        "Please select the language you want to generate the dialogue in. Currently, we support English and Chinese."
    )
    dialogue_language = gr.Radio(
        label="Dialogue Language",
        choices=["English", "Chinese"],
        value="English",
    )

    gr.Markdown("### Custom Prompt")
    current_tab = gr.State("input_prompt_tab")
    
    with gr.Tab("Input your custom prompt") as input_prompt_tab:
        with gr.Column():
            gr.Markdown("### Input your custom prompt")
            gr.Markdown("You can enter your custom prompt here. One prompt per line.")
            custom_prompt = gr.Textbox(
                label="Custom Prompt",
                placeholder="Enter your custom prompts here...",
                lines=10,
                max_lines=100,
            )

            @gr.render(inputs=custom_prompt)
            def count_prompts(text):
                num_prompt = (
                    len([u.strip() for u in text.split("\n") if len(u.strip()) > 0])
                    if text
                    else 0
                )
                gr.Textbox(
                    label="Number of Input Prompts",
                    value=num_prompt,
                    interactive=False,
                )

    with gr.Tab("Upload your custom prompt") as upload_prompt_tab:
        with gr.Column():
            gr.Markdown("### Upload your custom dialogue scenario")
            gr.Markdown(
                "You can upload a text file containing your custom dialogue scenario. The file should be in txt format, with one prompt per line."
            )
            custom_prompt_files = gr.File(label="Custom Prompt", file_types=[".txt"])

    input_prompt_tab.select(
        lambda: "input_prompt_tab",
        outputs=[current_tab],
    )
    upload_prompt_tab.select(
        lambda: "upload_prompt_tab",
        outputs=[current_tab],
    )
    
    num_dialogues_per_prompt = gr.Number(
        label="Number of Dialogues per Prompt",
        value=1,
        minimum=1,
        maximum=2000,
        interactive=True,
    )
    generate_btn = gr.Button(
        "Generate Command Line",
    )

    commandline_output = gr.Markdown("### Command Line Output")


    generate_btn.click(
        generate_command_line,
        inputs=[
            dialogue_language,
            custom_prompt,
            custom_prompt_files,
            num_dialogues_per_prompt,
            current_tab
        ],
        outputs=commandline_output,
    )
