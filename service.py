import bentoml
from PIL import Image as PILImage
import typing as tp 
from utils.utils import draw_detections, serialize_detections
from pydantic import BaseModel, Field
from pathlib import Path
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

runtime_image = bentoml.images.Image(
    python_version="3.11"
).pyproject_toml("pyproject.toml")

class DetectionParams(BaseModel):
    detection_prompt: tp.List[tp.List[str]] = Field(..., description="List of lists of labels, e.g. [['a cat', 'a remote control']]")
    box_threshold: float = Field(default=0.25, description="Box threshold between 0 and 1")
    text_threshold: float = Field(default=0.25, description="Text threshold between 0 and 1")
    
    class Config:
        arbitrary_types_allowed = True

@bentoml.service(
   name='grounding-dino-service',
   traffic={'timeout': 300},
   resources={'gpu': 1},
   workers=1,
   image=runtime_image
)
class GroundingDinoService:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_model).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.hf_model)
        print("Model grounding-dino loaded", "device:", self.device)

    @bentoml.api
    def detect_image(
        self,
        image: PILImage.Image,
        params: DetectionParams
        ) -> tp.List[tp.Dict[str, tp.Any]]:
        '''
        Detect objects in the image.
        '''
        return self._detect(image, params)

    def _detect(
        self,
        image: PILImage.Image,
        params: DetectionParams
    ) -> tp.List[tp.Dict[str, tp.Any]]:
        text_labels = params.detection_prompt
        inputs = self.processor(images=[image], text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=params.box_threshold,
            text_threshold=params.text_threshold,
            target_sizes=[(image.height, image.width)],
        )
        return serialize_detections(results)

    @bentoml.api
    def render(
        self,
        image: PILImage.Image,
        params: DetectionParams,
    ) -> PILImage.Image:
        '''
        Render detections on the image.
        '''
        result = self._detect(image, params)[0]
        image = draw_detections(image, result)
        Path("images").mkdir(exist_ok=True)
        image.save("images/out_render.jpg")
        return image
