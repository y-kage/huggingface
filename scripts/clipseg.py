import parser

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


def main(image, prompts, save_path="../results/clipseg.png"):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(
        text=prompts,
        images=[image] * len(prompts),
        padding="max_length",
        return_tensors="pt",
    )

    # predict
    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)

    num_preds = len(preds)
    num_cols = 5
    num_rows = (num_preds // num_cols) + 1

    _, ax = plt.subplots(
        num_rows + 1, num_cols, figsize=(num_cols * 3, (num_rows + 1) * 3)
    )

    [a.axis("off") for a in ax.flatten()]

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")

    for i in range(num_preds):
        row = (i // num_cols) + 1
        col = i % num_cols
        ax[row, col].imshow(torch.sigmoid(preds[i][0]).cpu().numpy())
        ax[row, col].set_title(prompts[i])

    for i in range(num_preds + num_cols, num_rows * num_cols):
        ax.flatten()[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)

    plt.savefig(save_path)


if __name__ == "__main__":

    # url = "https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image.save("../DATA/bowl.png")

    # text_prompt = ["a glass", "something to fill", "wood", "a jar"]
    # save_path = "../results/clipseg.png"

    parser = parser.parser()
    args = parser.parse_args()
    image_path = args.image_path
    save_path = args.save_path
    input_text = args.text_prompt

    if not image_path:
        image_path = "../DATA/bowl.png"
    if not input_text:
        input_text = ["a glass", "something to fill", "wood", "a jar"]
    if not save_path:
        save_path = "../results/clipseg.png"
    # print(args_dict)
    # if args.mode == "custom":
    #     image = Image.open(args.image_path)
    #     text_prompt = args.text_prompt
    #     save_path = args.save_path
    # print(text_prompt)

    image = Image.open(image_path)
    main(image, input_text, save_path)
