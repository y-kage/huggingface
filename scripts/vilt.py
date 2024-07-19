import requests
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

# prepare image + question
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("../DATA/000.jpg")
text = "What is in the image?"

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
print("Predicted answer:", model.config.id2label[idx])
