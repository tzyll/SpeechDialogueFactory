from ast import main
import gradio as gr

import app.sample_generation as main_page
import app.batch_generation as second_page
import app.sample_inspection as third_page
from argparse import ArgumentParser
parser = ArgumentParser()
main_page.add_arguments(parser=parser)

with gr.Blocks() as demo:
    main_page.demo.render()
with demo.route("Batch Generation"):
    second_page.demo.render()
with demo.route("Sample Inspection"):
    third_page.demo.render()

if __name__ == "__main__":
    demo.launch()