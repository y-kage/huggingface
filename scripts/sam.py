import os
import parser
import sys

import grounded_dino as bbox_detecter
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes, path="bbox.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.savefig(path)
    # plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None, path="point.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis("on")
    plt.savefig(path)
    # plt.show()


def show_points_and_boxes_on_image(
    raw_image, boxes, input_points, input_labels=None, path="point_bbox.png"
):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.savefig(path)
    # plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_masks_on_image(raw_image, masks, scores, path="mask.png"):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    plt.savefig(path)


def give_points(
    raw_image,
    input_points,
    labels=None,
    result_image_path="points_boxes.png",
    prompt_image_path="prompt.png",
):
    ## 1
    # input_points = [[[450, 600]]]  # 2D location of a window in the image
    if len(input_points[0]) == 1:
        show_input_points = input_points[0]
    else:
        # for i in range(len(input_points[0])):
        #     input_points[0][i] = [input_points[0][i]]
        # print("aaaaaa")
        show_input_points = input_points

    if not labels:
        labels = []
        for i in range(len(input_points[0])):
            labels.append(1)
        labels = [labels]
    if len(input_points[0]) == 1:
        show_label = labels[0]
    else:
        show_label = labels
    show_points_on_image(
        raw_image, show_input_points, show_label, path=prompt_image_path
    )
    if not len(input_points[0]) == 1:
        for i in range(len(input_points[0])):
            input_points[0][i] = [input_points[0][i]]
    if not len(input_points[0]) == 1:
        for i in range(len(input_points[0])):
            labels[0][i] = [labels[0][i]]

    inputs = processor(
        raw_image, input_points=input_points, input_labels=labels, return_tensors="pt"
    ).to(device)
    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    # show_masks_on_image(raw_image, masks[0], scores, path=result_image_path)
    parent_dir = os.path.dirname(result_image_path)
    file_name = os.path.basename(result_image_path).split(".", 1)[0]
    for i in range(len(masks[0])):
        result_image_path = f"{parent_dir}/{file_name}_{i}.png"
        show_masks_on_image(
            raw_image, masks[0][i], scores[:, 0, :], path=result_image_path
        )
        print(result_image_path)


def give_boxes(
    raw_image,
    input_boxes,
    result_image_path="points_boxes.png",
    prompt_image_path="prompt.png",
):

    show_boxes_on_image(raw_image, input_boxes[0], path=prompt_image_path)

    inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(
        device
    )
    inputs["input_boxes"].shape
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    # show_masks_on_image(raw_image, masks[0], scores, path=result_image_path)
    parent_dir = os.path.dirname(result_image_path)
    file_name = os.path.basename(result_image_path).split(".", 1)[0]
    for i in range(len(masks[0])):
        result_image_path = f"{parent_dir}/{file_name}_{i}.png"
        show_masks_on_image(
            raw_image, masks[0][i], scores[:, 0, :], path=result_image_path
        )
        print(result_image_path)


def give_points_boxes(
    raw_image,
    input_boxes,
    input_points,
    labels=None,
    result_image_path="points_boxes.png",
    prompt_image_path="prompt.png",
):
    if len(input_points[0]) == 1:
        show_input_points = input_points[0]
    else:
        show_input_points = input_points

    if len(input_boxes[0]) == 1:
        show_input = input_boxes[0]
    else:
        show_input = input_boxes

    if not labels:
        labels = []
        for i in range(len(input_points[0])):
            labels.append(1)
        labels = [labels]
    if len(input_points[0]) == 1:
        show_label = labels[0]
    else:
        show_label = labels

    show_points_and_boxes_on_image(
        raw_image, show_input, show_input_points, show_label, path=prompt_image_path
    )

    if not len(input_points[0]) == 1:
        for i in range(len(input_points[0])):
            input_points[0][i] = [input_points[0][i]]
    if not len(input_points[0]) == 1:
        for i in range(len(input_points[0])):
            labels[0][i] = [labels[0][i]]
    inputs = processor(
        raw_image,
        input_boxes=[input_boxes],
        input_points=input_points,
        input_labels=labels,
        return_tensors="pt",
    ).to(device)

    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    # print(len(masks[0]))
    parent_dir = os.path.dirname(result_image_path)
    file_name = os.path.basename(result_image_path).split(".", 1)[0]
    for i in range(len(masks[0])):
        result_image_path = f"{parent_dir}/{file_name}_{i}.png"
        show_masks_on_image(
            raw_image, masks[0][i], scores[:, 0, :], path=result_image_path
        )
        print(result_image_path)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # img_url = (
    #     "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # )
    # img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    raw_image.save("../DATA/dog.png")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    raw_image.save("../DATA/dog_mask.png")

    parser = parser.parser()
    args = parser.parse_args()
    mode = args.mode
    if not mode:
        mode = "points"
    image_path = args.image_path
    result_image_path = args.save_path
    input_points = args.points_prompt
    input_boxes = args.boxes_prompt
    input_labels = args.labels_prompt
    input_text = args.text_prompt

    if not image_path:
        image_path = "../DATA/dog.png"
    if not args.save_path:
        result_image_path = "../results/sam.png"

    raw_image = Image.open(image_path)
    inputs = processor(raw_image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    prompt_image_path = "../results/prompt.png"

    if mode == "points":
        if not input_points:
            input_points = [[[850, 1100], [2250, 1000]]]
        give_points(
            raw_image,
            input_points,
            labels=input_labels,
            result_image_path=result_image_path,
            prompt_image_path=prompt_image_path,
        )
    elif mode == "boxes":
        if not input_boxes:
            input_boxes = [[[650, 900, 1000, 1250], [2050, 800, 2400, 1150]]]
        give_boxes(
            raw_image,
            input_boxes,
            result_image_path=result_image_path,
            prompt_image_path=prompt_image_path,
        )
    elif mode == "points_boxes":
        if not input_points:
            input_points = [[[850, 1100], [2250, 1000]]]
        if not input_boxes:
            input_boxes = [[[650, 900, 1000, 1250], [2050, 800, 2400, 1150]]]
        give_points_boxes(
            raw_image,
            input_boxes,
            input_points,
            labels=input_labels,
            result_image_path=result_image_path,
            prompt_image_path=prompt_image_path,
        )
    elif mode == "grounded_dino":
        if not input_text:
            input_text = "a cat. a remote control."
        bbox = bbox_detecter.main(
            image=raw_image, text=input_text, save_path="../results/sam_bbox_detect.png"
        )
        if len(bbox["labels"]) == 0:
            print("No bbox")
        else:
            input_boxes = [bbox["boxes"].tolist()]
            give_boxes(
                raw_image,
                input_boxes,
                result_image_path=result_image_path,
                prompt_image_path=prompt_image_path,
            )
