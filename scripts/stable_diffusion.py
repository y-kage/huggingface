import os
import parser

from diffusers import DiffusionPipeline


def main(text, save_dir):
    generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    generator.to("cuda")
    images = generator(text).images
    # for i, img in enumerate(images):
    #     img.save(os.path.join(save_dir, f"stable_diffusion_{i}.png"))
    images[0].save(os.path.join(save_dir, f"stable_diffusion.png"))


if __name__ == "__main__":

    parser = parser.parser()
    args = parser.parse_args()
    text = args.text_prompt
    save_path = args.save_path
    if not text:
        text = "An image of a squirrel in Picasso style"
    if not save_path:
        save_path = "../results"

    main(text, save_path)
