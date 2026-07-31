"""Gradio interface combining MegaDetector with a custom classification model."""

import torch
import numpy as np
import gradio as gr
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from wildlife_classifier import (
    WildlifeClassificationAdapter,
    format_species_detection_label,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Annotators for visualising detections
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=4)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

DETECTION_MODEL_PATH = "MegaDetectorV6.pt"
CLASSIFICATION_MODEL_PATH = "wildlife_model.pth"

print("Loading MegaDetector model...")
detection_model = YOLO(DETECTION_MODEL_PATH)
print("MegaDetector loaded successfully.")

print("Loading classification model...")
classification_model = WildlifeClassificationAdapter(CLASSIFICATION_MODEL_PATH, device=DEVICE)
print("Classification model loaded successfully.")

def classify_species(image: Image.Image) -> dict:
    """Predict species for a crop."""
    return classification_model.single_image_classification(image)


def single_image_detection(
    input_img: Image.Image,
    det_conf_thres: float = 0.25,
    clf_conf_thres: float = 0.70,
) -> Image.Image:
    """Run detection and classification on a single image."""
    input_img_np = np.array(input_img)
    raw_result = detection_model(input_img_np)[0]
    detections = sv.Detections.from_ultralytics(raw_result)

    if len(detections) == 0:
        return input_img

    mask = detections.confidence > det_conf_thres
    filtered_detections = detections[mask]

    if len(filtered_detections) == 0:
        return input_img

    labels = []
    for bbox, confidence in zip(filtered_detections.xyxy, filtered_detections.confidence):
        x_min, y_min, x_max, y_max = map(int, bbox)
        crop = input_img.crop((x_min, y_min, x_max, y_max))
        species = classify_species(crop)
        labels.append(format_species_detection_label(species, conf_threshold=clf_conf_thres))

    annotated_scene = LABEL_ANNOTATOR.annotate(
        scene=BOX_ANNOTATOR.annotate(scene=input_img_np.copy(), detections=filtered_detections),
        detections=filtered_detections,
        labels=labels,
    )
    return Image.fromarray(annotated_scene)


with gr.Blocks() as demo:
    gr.Markdown("# MegaDetector + Classification Model")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload image")
        conf_threshold = gr.Slider(0, 1, value=0.25, label="Detection confidence threshold")
        clf_conf_threshold = gr.Slider(0, 1, value=0.70, label="Classification confidence threshold")

    img_output = gr.Image(label="Annotated image")
    detect_button = gr.Button("Detect animals")

    detect_button.click(
        single_image_detection,
        inputs=[img_input, conf_threshold, clf_conf_threshold],
        outputs=img_output,
    )


if __name__ == "__main__":
    demo.launch(share=True)

