import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import ViltForQuestionAnswering, ViltProcessor

try:
    from . import global_parser
except ImportError:
    import global_parser


def add_white_margin(image, margin_height):
    width, height = image.size
    new_height = height + margin_height
    new_image = Image.new("RGB", (width, new_height), "white")
    new_image.paste(image, (0, 0))
    return new_image


def draw_caption(image, caption, margin_height=100):
    image = add_white_margin(image, margin_height)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf", 40
        )  # Change the font size if needed
    except IOError:
        font = ImageFont.load_default()

    # Get text size using textbbox (Bounding Box)
    text_bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    width, height = image.size
    text_x = (width - text_width) / 2
    text_y = height - margin_height + (margin_height - text_height) / 2

    draw.text((text_x, text_y), caption, font=font, fill="black")
    return image


def main(image, text, save_path):

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    # print(type(logits))
    # print(logits.shape)
    # print(logits.argmax(-1))
    # print("Predicted answer:", model.config.id2label[idx])
    caption = model.config.id2label[idx]
    caption = f"Predicted answer: {caption}"
    print(caption)
    # Draw the caption on the image
    image_with_caption = draw_caption(image, caption)

    # Save the image with the caption
    image_with_caption.save(save_path)
    print(f"Image saved with caption to {save_path}")


if __name__ == "__main__":
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image.save("../DATA/cat.png")

    global_parser = global_parser.parser()
    args = global_parser.parse_args()
    image_path = args.image_path
    save_path = args.save_path
    input_text = args.text_prompt

    if not image_path:
        image_path = "../DATA/cat.png"
    if not input_text:
        input_text = "What is in the image?"
    if not save_path:
        save_path = "../results/vilt.png"

    image = Image.open(image_path)
    main(image, input_text, save_path=save_path)
