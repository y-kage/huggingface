import parser
import random

import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main(image, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    vis_bbox(image, results)
    # print(results)
    return results


def vis_bbox(image, results):
    if results:
        for i in range(results["scores"].shape[0]):
            _bbox = results["boxes"][i]
            bbox = []
            for v in _bbox:
                bbox.append(int(v))
            x_min, y_min, x_max, y_max = bbox
            bbox = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            draw = ImageDraw.Draw(image)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            draw.polygon(bbox, outline=color, width=2)

        output_path = "../results/test_bbox.jpg"
        image.save(output_path)


if __name__ == "__main__":
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    text = "a cat. a remote control."
    parser = parser.parser()
    args = parser.parse_args()
    if args.image_path:
        image = Image.open(args.image_path)
    if args.text_prompt:
        text = args.text_prompt

    main(image=image, text=text)
