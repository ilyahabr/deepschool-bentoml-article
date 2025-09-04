from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image: Image.Image, result: Dict[str, Any], color: str = "lime", box_width: int = 3) -> Image.Image:
	"""
	Draw bounding boxes and labels on a copy of `image` using a serialized Grounding DINO result dict.
	Expects keys: 'boxes' (list[list[float]]), 'scores' (list[float]), 'text_labels' (list[str]).
	"""
	img = image.copy()
	draw = ImageDraw.Draw(img)
	try:
		font = ImageFont.truetype("DejaVuSans.ttf", 14)
	except Exception:
		font = ImageFont.load_default()

	boxes = result["boxes"]
	scores = result["scores"]
	labels = result["text_labels"]

	for box, score, label in zip(boxes, scores, labels):
		x0, y0, x1, y1 = map(float, box)
		draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=box_width)

		text = f"{label} {float(score):.2f}"
		tb = draw.textbbox((0, 0), text, font=font)
		tw, th = tb[2] - tb[0], tb[3] - tb[1]
		ty = max(0, y0 - th - 2)
		draw.rectangle([x0, ty, x0 + tw + 4, ty + th + 2], fill=color)
		draw.text((x0 + 2, ty + 1), text, fill="black", font=font)

	return img

# --- new: JSON-safe serialization for Grounding DINO results ---
from typing import List, Sequence

def serialize_detections(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Convert Grounding DINO post-processed results into JSON-serializable structures.
	- boxes: list[list[float]]
	- scores: list[float]
	- text_labels: list[str]
	"""
	serializable: List[Dict[str, Any]] = []
	for r in results:
		boxes = r.get("boxes")
		scores = r.get("scores")
		labels = r.get("text_labels") or r.get("labels") or []

		# boxes -> list[list[float]]
		if boxes is None:
			boxes_out = []
		else:
			boxes_out = boxes.tolist() if hasattr(boxes, "tolist") else boxes
			boxes_out = [[float(v) for v in b] for b in boxes_out]

		# scores -> list[float]
		if scores is None:
			scores_out = []
		else:
			scores_out = scores.tolist() if hasattr(scores, "tolist") else scores
			scores_out = [float(s) for s in scores_out]

		labels_out = [str(l) for l in labels]
		serializable.append({"boxes": boxes_out, "scores": scores_out, "text_labels": labels_out})

	return serializable



