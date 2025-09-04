import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.utils import draw_detections

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda:1"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text_labels = [["a cat", "a remote control"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[(image.height, image.width)]
)
# Retrieve the first image result
print(results)
result = results[0]
for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
# vis = draw_detections(image, result) 
# vis.save("images/out_demo.jpg")