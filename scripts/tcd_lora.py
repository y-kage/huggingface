import os
import sys

import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline, TCDScheduler
from diffusers.utils import load_image, make_image_grid
from PIL import Image

try:
    from . import global_parser
except ImportError:
    import global_parser


def text_2_image(text, save_dir):
    device = "cuda"
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    images = pipe(
        prompt=text,
        num_inference_steps=4,
        guidance_scale=0,
        eta=0.3,
        generator=torch.Generator(device=device).manual_seed(0),
    ).images
    # for i, img in enumerate(images):
    #     img.save(os.path.join(save_dir, f"tcd_lora_T2I_{i}.png"))
    images[0].save(os.path.join(save_dir, f"tcd_lora_T2I.png"))


def inpainting(image, mask, text, save_dir):
    device = "cuda"
    base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    pipe = AutoPipelineForInpainting.from_pretrained(
        base_model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    init_image = image.resize((1024, 1024))
    mask_image = mask.resize((1024, 1024))

    prompt = text

    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=8,
        guidance_scale=0,
        eta=0.3,
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]

    grid_image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    # print(type(grid_image))
    grid_image.save(os.path.join(save_dir, f"tcd_lora_inpainting.png"))


if __name__ == "__main__":

    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    # init_image = load_image(img_url).resize((1024, 1024))
    # init_image.save("../DATA/dog.png")
    # mask_image = load_image(mask_url).resize((1024, 1024))

    global_parser = global_parser.parser()
    args = global_parser.parse_args()
    mode = args.mode
    if not mode:
        mode = "text2image"
    text = args.text_prompt
    save_path = args.save_path
    image_path = args.image_path
    mask_path = args.mask_path
    if not text:
        # text = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."
        text = "a frog sitting on a park bench"
    if not save_path:
        save_path = "../results"
    if not image_path:
        image_path = "../DATA/dog.png"
    if not mask_path:
        mask_path = "../DATA/dog_mask.png"

    if mode == "inpainting":
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        inpainting(image, mask, text, save_path)
    else:  # mode == "text2image":
        text_2_image(text, save_path)
