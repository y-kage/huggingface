import ast

import gradio as gr
import scripts

# points_list = []
# labels_list = []
with gr.Blocks() as demo:
    points_list = gr.State(value=[])
    boxes_list = gr.State(value=[])
    labels_list = gr.State(value=[])
    selected_list = gr.State(value=[])
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            mode = gr.Dropdown(
                choices=["points", "boxes"],
                value="points",
                label="Prompt type",
            )
            selected_points = gr.State([])
            selected_label = gr.Radio(
                choices=[("foreground", 1), ("background", 0)],
                value=1,
                label="Label",
                info="Is the point foreground or background of segmentation?",
            )
            output_text = gr.Textbox(lines=1, label="Selected Point")
            add_button = gr.Button("Add")
        with gr.Column():
            prompt_img = gr.Image(type="pil")
            point_text = gr.Textbox(lines=1, label="Prompt Points")
            box_text = gr.Textbox(lines=1, label="Prompt Boxes")
            label_text = gr.Textbox(lines=1, label="Prompt Lables")
            clear_button = gr.Button("Clear all Points")
            segment_button = gr.Button("Run Segmentation")
        with gr.Column():
            segment_img = gr.Image(type="pil")

    def click_handler(image, mode, selected_list, event: gr.SelectData):
        # print("Click event:", event.index)
        if event is not None:
            x, y = event.index[0], event.index[1]
            # print(f"Clicked at ({x}, {y})")
            if x is not None and y is not None:
                selected_list.append([x, y])
                # print(selected_list)
                if len(selected_list) > 1:
                    if mode == "points":
                        selected_list = [selected_list[-1]]
                    elif mode == "boxes":
                        selected_list = selected_list[-2:]
                # print(selected_list)
                return selected_list, selected_list
            # return ("", "")
        # return ("", "")
        return ""

    def add_button_handler(
        points_list,
        boxes_list,
        labels_list,
        raw_image,
        new_point,
        new_label,
        prompt_img,
    ):
        # global points_list
        # global labels_list
        if new_point == "":
            print("No new prompt")
        else:
            # new_point = ast.literal_eval(new_point)
            # print(type(new_point))
            # print(new_point)
            if len(new_point) == 1:
                points_list.append(new_point[0])
                labels_list.append(new_label)
                prompt_img = scripts.sam.show_points_on_image(
                    raw_image, points_list, input_labels=labels_list
                )
            elif len(new_point) == 2:
                _new_list = []
                _new_list.append(min(new_point[0][0], new_point[1][0]))
                _new_list.append(min(new_point[0][1], new_point[1][1]))
                _new_list.append(max(new_point[0][0], new_point[1][0]))
                _new_list.append(max(new_point[0][1], new_point[1][1]))
                # print(_new_list)
                boxes_list.append(_new_list)
                # labels_list.append(new_label)
                prompt_img = scripts.sam.show_boxes_on_image(raw_image, boxes_list)
        return (
            points_list,
            boxes_list,
            labels_list,
            prompt_img,
            points_list,
            boxes_list,
            labels_list,
        )

    def clear_button_handler(points_list, boxes_list, labels_list, raw_image):
        # global points_list
        # global labels_list
        points_list = []
        boxes_list = []
        labels_list = []
        return (
            points_list,
            boxes_list,
            labels_list,
            raw_image,
            points_list,
            boxes_list,
            labels_list,
        )

    input_image.select(
        click_handler,
        inputs=[input_image, mode, selected_list],
        outputs=[output_text, selected_list],
    )
    add_button.click(
        fn=add_button_handler,
        inputs=[
            points_list,
            boxes_list,
            labels_list,
            input_image,
            selected_list,
            selected_label,
            prompt_img,
        ],
        outputs=[
            points_list,
            boxes_list,
            labels_list,
            prompt_img,
            point_text,
            box_text,
            label_text,
        ],
    )
    clear_button.click(
        fn=clear_button_handler,
        inputs=[points_list, boxes_list, labels_list, input_image],
        outputs=[
            points_list,
            boxes_list,
            labels_list,
            prompt_img,
            point_text,
            box_text,
            label_text,
        ],
    )

    segment_button.click(
        fn=scripts.sam.main,
        inputs=[
            mode,
            input_image,
            points_list,
            boxes_list,
            labels_list,
            point_text,
        ],
        outputs=[prompt_img, segment_img],
    )

demo.launch(server_name="0.0.0.0", server_port=8869)
