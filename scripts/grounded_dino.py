import parser
import random

import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main(image, text, save_path):
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
    vis_bbox(image, results, save_path)
    # print(results)
    return results


def vis_bbox(image, results, save_path):
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

        image.save(save_path)


if __name__ == "__main__":
    parser = parser.parser()
    args = parser.parse_args()
    image_path = args.image_path
    text = args.text_prompt
    save_path = args.save_path
    if not image_path:
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_path = requests.get(image_url, stream=True).raw
    if not text:
        text = "a cat. a remote control."
    if not save_path:
        save_path = "../results/grounded_dino.jpg"

    image = Image.open(image_path)
    main(image=image, text=text, save_path=save_path)
