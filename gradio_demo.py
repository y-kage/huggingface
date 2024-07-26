import importlib
import os
import random

# import clipseg
# import depth_anything
import gradio as gr
import numpy as np
import scripts

if __name__ == "__main__":
    """
    def random_response(image: np.array, message, history):
        return random.choice(["Yes", "No"])


    def greet(name, intensity):
        return "Hello, " + name + "!" * int(intensity)


    demo = gr.Interface(
        fn=greet,
        inputs=[
            gr.Image(type="numpy", label="Upload an image"),
            gr.Textbox(label="Enter some text", value="test input text"),
            gr.Slider(minimum=0, maximum=100, value=50, label="Adjust the slider"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Processed Image"),
            gr.Textbox(label="Resulting Text"),
        ],
    )

    # demo.launch()
    # demo.launch(share=True) # To share
    demo.launch(server_name="0.0.0.0", server_port=8869)
    """
    # # or by in chat format
    # gr.ChatInterface(random_response).launch()
    # gr.ChatInterface(random_response).launch(server_name="0.0.0.0", server_port=8869)

    # def hello(name: str) -> str:
    #     return f"Hello, {name}!"

    # with gr.Blocks() as hello_blocks_interface:
    #     input = gr.Textbox(label="Name")
    #     output = gr.Textbox(label="Output Box")
    #     button = gr.Button("Hello")
    #     button.click(fn=hello, inputs=input, outputs=output, api_name="hello")

    # hello_blocks_interface.launch(server_name="0.0.0.0", server_port=8869)

    """
    with gr.Blocks() as calculator_blocks:
        with gr.Tab("Sqrt"):
            x = gr.Number(label="x")
            sqrt_button = gr.Button("Calculate")
        with gr.Tab("Add"):
            with gr.Row():
                a = gr.Number(label="a")
                b = gr.Number(label="b")
            add_button = gr.Button("Calculate")

        output = gr.Number(label="Result")

        sqrt_button.click(fn=lambda x: x**0.5, inputs=x, outputs=output)
        add_button.click(fn=lambda a, b: a + b, inputs=[a, b], outputs=output)

    calculator_blocks.launch(server_name="0.0.0.0", server_port=8869)
    """

    with gr.Blocks() as ml_blocks:
        with gr.Tab("Clipseg"):
            with gr.Row():
                with gr.Column():
                    clipseg_img = gr.Image(type="pil", label="Upload an image")
                    clipseg_text = gr.Textbox(
                        label="Enter some text", value="test input text"
                    )
                    # x = gr.Number(label="x")
                    clipseg_button = gr.Button("Run")
                with gr.Column():
                    clipseg_output = gr.Image(type="pil")
            clipseg_button.click(
                fn=scripts.clipseg.main,
                inputs=[clipseg_img, clipseg_text],
                outputs=clipseg_output,
            )

        with gr.Tab("Depth Anything"):
            with gr.Row():
                with gr.Column():
                    depth_anything_img = gr.Image(type="pil", label="Upload an image")
                    depth_anything_button = gr.Button("Run")
                with gr.Column():
                    depth_anything_output = gr.Image(type="pil")

            depth_anything_button.click(
                fn=scripts.depth_anything.main,
                inputs=[depth_anything_img],
                outputs=depth_anything_output,
            )
        with gr.Tab("Segment Anything Model (SAM)"):
            with gr.Row():
                with gr.Column():
                    sam_mode = gr.Dropdown(
                        choices=["text", "points", "boxes", "points_boxes"],
                        value="text",
                        label="Prompt type",
                    )
                    sam_img = gr.Image(type="pil", label="Upload an image")
                    sam_points = gr.Textbox(
                        label="Enter some text when mode=='points' or 'points_boxes'",
                        value=None,
                    )
                    sam_labels = gr.Textbox(
                        label="Enter some text when mode=='points' or 'points_boxes'",
                        value=None,
                    )
                    sam_boxes = gr.Textbox(
                        label="Enter some text when mode=='boxes' or 'points_boxes'",
                        value=None,
                    )
                    sam_text = gr.Textbox(
                        label="Enter some text when mode=='text'",
                        value="a door. ground.",
                    )
                    # x = gr.Number(label="x")
                    sam_button = gr.Button("Run")
                with gr.Column():
                    sam_prompt = gr.Image(type="pil")
                    sam_output = gr.Image(type="pil")
            sam_button.click(
                fn=scripts.sam.main,
                inputs=[sam_mode, sam_img, sam_points, sam_boxes, sam_labels, sam_text],
                outputs=[sam_prompt, sam_output],
            )

    ml_blocks.launch(server_name="0.0.0.0", server_port=8869)
