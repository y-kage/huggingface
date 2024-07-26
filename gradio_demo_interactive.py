import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy")
            selected_points = gr.State([])
        with gr.Column():
            output_text = gr.Textbox(lines=1, label="Output")

        def click_handler(image, event: gr.SelectData):
            print("Click event:", event.index)
            if event is not None:
                x, y = event.index[0], event.index[1]
                print(f"Clicked at ({x}, {y})")
                if x is not None and y is not None:
                    return (x, y)
                return ("", "")
            return ("", "")

        input_image.select(click_handler, inputs=[input_image], outputs=[output_text])

demo.launch(server_name="0.0.0.0", server_port=8869)
