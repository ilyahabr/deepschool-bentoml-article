import bentoml
import requests
from io import BytesIO
from PIL import Image as PILImage

img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = PILImage.open(BytesIO(requests.get(img_url, stream=True).content))

with bentoml.SyncHTTPClient("http://localhost:3025") as client:
    result = client.detect_image(
        image=img,
        params={
            "detection_prompt": [["a cat", "a remote control"]],
            "box_threshold": 0.25,
            "text_threshold": 0.25,
        },
    )
    print(result)
