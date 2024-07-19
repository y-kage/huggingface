import parser

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main(image):

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
    depth.save("../results/depth.png")
    return depth


if __name__ == "__main__":

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image.save("../DATA/cat.png")

    parser = parser.parser()
    args = parser.parse_args()
    raw_image = Image.open(args.image_path)

    main(raw_image)
