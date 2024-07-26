import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

try:
    from . import global_parser
except ImportError:
    import global_parser


def main(image, prompts, save_flag=False, save_path="../results/clipseg.png"):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    prompts = prompts.split(".")
    new_prompts = []
    for text in prompts:
        if text == "":
            continue
        if text[0] == " ":
            new_prompts.append(text[1:])
        else:
            new_prompts.append(text)
    prompts = new_prompts

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

    fig, ax = plt.subplots(
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

    # Convert figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    # Convert RGBA to RGB if needed
    img_array = img_array[..., :3]

    if save_flag:
        plt.imsave(save_path, img_array)

    plt.close(fig)
    return img_array


if __name__ == "__main__":

    # url = "https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image.save("../DATA/bowl.png")

    # text_prompt = ["a glass", "something to fill", "wood", "a jar"]
    # save_path = "../results/clipseg.png"

    global_parser = global_parser.parser()
    args = global_parser.parse_args()
    image_path = args.image_path
    save_path = args.save_path
    input_text = args.text_prompt

    if not image_path:
        image_path = "../DATA/bowl.png"
    if not input_text:
        input_text = "a glass. something to fill. wood. a jar"
    if not save_path:
        save_path = "../results/clipseg.png"
    # print(args_dict)
    # if args.mode == "custom":
    #     image = Image.open(args.image_path)
    #     text_prompt = args.text_prompt
    #     save_path = args.save_path
    # print(text_prompt)

    image = Image.open(image_path)
    main(image, input_text, save_flag=True, save_path=save_path)
