import parser

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main(image, save_path):

    image_processor = AutoImageProcessor.from_pretrained(
        "LiheYoung/depth-anything-small-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-small-hf"
    )

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save(save_path)
    return depth


if __name__ == "__main__":

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image.save("../DATA/cat.png")

    parser = parser.parser()
    args = parser.parse_args()
    image_path = args.image_path
    save_path = args.save_path
    if not args.image_path:
        image_path = "../DATA/cat.png"
    if not args.save_path:
        save_path = "../results/depth_anything.png"

    image = Image.open(image_path)
    main(image, save_path=save_path)
