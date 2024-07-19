import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

url = "https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)


prompts = ["a glass", "something to fill", "wood", "a jar"]

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

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis("off") for a in ax.flatten()]
ax[0].imshow(image)
[ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
[ax[i + 1].text(0, -15, prompts[i]) for i in range(4)]
