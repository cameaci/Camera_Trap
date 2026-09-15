# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Gradio Demo for image detection"""

import os
import torch
import shutil
import cv2
import supervision as sv
import gradio as gr
from zipfile import ZipFile
from torch.utils.data import DataLoader
import numpy as np
import ast


def patch_yolov5_scale_coords():
    """Keep PytorchWildlife 1.2.0 compatible with newer yolov5 packages."""
    try:
        from yolov5.utils import general as yolov5_general
    except Exception:
        return

    if not hasattr(yolov5_general, "scale_coords") and hasattr(yolov5_general, "scale_boxes"):
        yolov5_general.scale_coords = yolov5_general.scale_boxes


patch_yolov5_scale_coords()

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data
from wildlife_classifier import (
    WildlifeClassificationAdapter,
    format_species_detection_label,
)
from speciesnet_adapter import SpeciesNetClassificationAdapter

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Annotators
dot_annotator = sv.DotAnnotator(radius=6)
box_annotator = sv.BoxAnnotator(thickness=4)

# Ensure temp folder exists
os.makedirs(os.path.join("..", "temp"), exist_ok=True)

# Global models
detection_model = None
classification_model = None

WILDLIFE_MODEL_PATH = "wildlife_model.pth"


def short_error(exc):
    return str(exc).splitlines()[0]


def detection_class_name(det_id):
    class_names = getattr(detection_model, "CLASS_NAMES", {})
    try:
        det_id = int(det_id)
    except (TypeError, ValueError):
        return ""

    if isinstance(class_names, dict):
        return str(class_names.get(det_id, class_names.get(str(det_id), "")))
    if isinstance(class_names, (list, tuple)) and 0 <= det_id < len(class_names):
        return str(class_names[det_id])
    return ""


def is_animal_detection(det_id, default_label):
    candidates = [
        detection_class_name(det_id),
        str(default_label).split()[0] if default_label is not None else "",
    ]
    return any(candidate.strip().lower() == "animal" for candidate in candidates)


def load_models(det, version, clf, wpath=None, wclass=None):
    global detection_model, classification_model
    classifier_status = "No classifier"

    # Load detection model
    if det != "None":
        if det == "HerdNet General":
            detection_model = pw_detection.HerdNet(device=DEVICE)
        elif det == "HerdNet Ennedi":
            detection_model = pw_detection.HerdNet(device=DEVICE, version="ennedi")
        else:
            detection_model = pw_detection.__dict__[det](
                device=DEVICE, pretrained=True, version=version
            )
    else:
        detection_model = None
        return "NO MODEL LOADED!!"

    # Load classification model
    if clf != "None":
        if clf == "CustomWeights":
            try:
                if not wpath or not wclass:
                    raise ValueError("CustomWeights requires both weights path and class mapping.")
                wclass = ast.literal_eval(wclass)
                classification_model = pw_classification.__dict__[clf](
                    weights=wpath, class_names=wclass, device=DEVICE
                )
                classifier_status = "CustomWeights"
            except Exception as e:
                print(f"CustomWeights could not be loaded: {e}")
                classification_model = None
                classifier_status = f"CustomWeights failed: {short_error(e)}"
        elif clf == "WildlifeModel":
            try:
                weights_path = wpath.strip() if isinstance(wpath, str) and wpath.strip() else WILDLIFE_MODEL_PATH
                classification_model = WildlifeClassificationAdapter(
                    weights_path=weights_path,
                    device=DEVICE,
                )
                print(f"Wildlife model successfully loaded from {weights_path}.")
                recommended_thr = classification_model.metrics.get("recommended_clf_conf_threshold")
                if isinstance(recommended_thr, (float, int)):
                    print(f"Recommended classification threshold: {float(recommended_thr):.2f}")
                classifier_status = f"WildlifeModel ({weights_path})"
            except Exception as e:
                print(f"Wildlife model could not be loaded: {e}")
                classification_model = None
                classifier_status = f"WildlifeModel failed: {short_error(e)}"
        elif clf == "SpeciesNet":
            try:
                model_name = wpath.strip() if isinstance(wpath, str) and wpath.strip() else None
                classification_model = SpeciesNetClassificationAdapter(
                    model_name=model_name,
                    device=DEVICE,
                )
                print(f"SpeciesNet successfully loaded from {classification_model.model_name}.")
                classifier_status = f"SpeciesNet ({classification_model.model_name})"
            except Exception as e:
                print(f"SpeciesNet could not be loaded: {e}")
                classification_model = None
                classifier_status = f"SpeciesNet failed: {short_error(e)}"
        else:
            try:
                classification_model = pw_classification.__dict__[clf](
                    device=DEVICE, pretrained=True
                )
                classifier_status = clf
            except Exception as e:
                print(f"{clf} could not be loaded: {e}")
                classification_model = None
                classifier_status = f"{clf} failed: {short_error(e)}"
    else:
        classification_model = None

    return f"Loaded Detector: {det}. Version: {version}. Classifier: {classifier_status}"

def single_image_detection(input_img, det_conf_thres, clf_conf_thres, img_index=None):
    img = np.array(input_img)
    # Choose annotator & run detection
    if "HerdNet" in detection_model.__class__.__name__:
        annotator = dot_annotator
        results_det = detection_model.single_image_detection(
            img, img_path=img_index,
            det_conf_thres=det_conf_thres,
            clf_conf_thres=clf_conf_thres
        )
    else:
        annotator = box_annotator
        results_det = detection_model.single_image_detection(
            img, img_path=img_index,
            det_conf_thres=det_conf_thres
        )

    default_labels = list(results_det["labels"])
    if classification_model and hasattr(classification_model, "single_image_classification"):
        labels = []
        for xyxy, det_id, default_label in zip(
            results_det["detections"].xyxy,
            results_det["detections"].class_id,
            default_labels,
        ):
            if is_animal_detection(det_id, default_label):
                crop = sv.crop_image(image=img, xyxy=xyxy)
                try:
                    res_clf = classification_model.single_image_classification(crop)
                    labels.append(
                        format_species_detection_label(
                            res_clf,
                            conf_threshold=clf_conf_thres,
                        )
                    )
                except Exception as e:
                    print(f"Classification failed for detection {det_id}: {e}")
                    labels.append(default_label)
            else:
                labels.append(default_label)
    else:
        labels = default_labels

    annotated = annotator.annotate(scene=img, detections=results_det["detections"])
    return draw_labels_within_frame(annotated, results_det["detections"], labels)


def draw_labels_within_frame(scene, detections, labels):
    """Draw labels while clamping them to image boundaries."""
    if len(labels) == 0:
        return scene

    annotated = scene.copy()
    img_h, img_w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(0.45, min(0.80, img_w / 1400.0))
    text_thickness = 1 if img_w < 1600 else 2
    padding = 4
    max_chars = 22

    for xyxy, label in zip(detections.xyxy, labels):
        x_min, y_min, _, _ = map(int, xyxy)
        text = str(label)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."

        (text_w, text_h), baseline = cv2.getTextSize(text, font, text_scale, text_thickness)
        box_w = text_w + (padding * 2)
        box_h = text_h + baseline + (padding * 2)

        x_box = min(max(0, x_min), max(0, img_w - box_w - 1))
        y_box = y_min - box_h - 4
        if y_box < 0:
            y_box = y_min + 4
        y_box = min(max(0, y_box), max(0, img_h - box_h - 1))

        top_left = (x_box, y_box)
        bottom_right = (x_box + box_w, y_box + box_h)
        cv2.rectangle(annotated, top_left, bottom_right, (255, 64, 160), thickness=-1)

        text_x = x_box + padding
        text_y = y_box + padding + text_h
        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            font,
            text_scale,
            (0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated

def batch_detection(zip_file, timelapse, det_conf_thres):
    extract_path = os.path.join("..","temp","zip_upload")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    json_save = os.path.join(extract_path, "results.json")
    with ZipFile(zip_file.name) as z:
        z.extractall(extract_path)

    files = os.listdir(extract_path)
    tgt = os.path.join(extract_path, files[0]) if len(files)==1 and os.path.isdir(os.path.join(extract_path, files[0])) else extract_path

    if "HerdNet" in detection_model.__class__.__name__:
        det_res = detection_model.batch_image_detection(tgt, batch_size=1, det_conf_thres=det_conf_thres, id_strip=tgt)
    else:
        det_res = detection_model.batch_image_detection(tgt, batch_size=16, det_conf_thres=det_conf_thres, id_strip=tgt)

    classifier_supports_batch = (
        classification_model
        and hasattr(classification_model, "batch_image_classification")
        and callable(getattr(classification_model, "batch_image_classification"))
    )

    if classifier_supports_batch:
        dataset = pw_data.DetectionCrops(
            det_res,
            transform=pw_trans.Classification_Inference_Transform(target_size=224),
            path_head=tgt
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        clf_res = classification_model.batch_image_classification(loader, id_strip=tgt)
        clf_categories = getattr(classification_model, "CLASS_NAMES", {})

        if timelapse:
            json_save = json_save.replace(".json","_timelapse.json")
            pw_utils.save_detection_classification_timelapse_json(
                det_results=det_res, clf_results=clf_res,
                det_categories=detection_model.CLASS_NAMES,
                clf_categories=clf_categories,
                output_path=json_save
            )
        else:
            pw_utils.save_detection_classification_json(
                det_results=det_res, clf_results=clf_res,
                det_categories=detection_model.CLASS_NAMES,
                clf_categories=clf_categories,
                output_path=json_save
            )
    else:
        if classification_model and not classifier_supports_batch:
            print("Loaded classification model does not support batch processing; skipping classification.")
        if timelapse:
            json_save = json_save.replace(".json","_timelapse.json")
            pw_utils.save_detection_timelapse_json(det_res, json_save, categories=detection_model.CLASS_NAMES)
        elif "HerdNet" in detection_model.__class__.__name__:
            pw_utils.save_detection_json_as_dots(det_res, json_save, categories=detection_model.CLASS_NAMES)
        else:
            pw_utils.save_detection_json(det_res, json_save, categories=detection_model.CLASS_NAMES)

    return json_save

def batch_path_detection(tgt_folder_path, det_conf_thres):
    json_save = os.path.join(tgt_folder_path, "results.json")
    det_res = detection_model.batch_image_detection(tgt_folder_path, det_conf_thres=det_conf_thres, id_strip=tgt_folder_path)
    if "HerdNet" in detection_model.__class__.__name__:
        pw_utils.save_detection_json_as_dots(det_res, json_save, categories=detection_model.CLASS_NAMES)
    else:
        pw_utils.save_detection_json(det_res, json_save, categories=detection_model.CLASS_NAMES)
    return json_save


def video_detection(video, det_conf_thres, clf_conf_thres, target_fps, codec):
    def cb(frame, i):
        return single_image_detection(frame, det_conf_thres, clf_conf_thres, img_index=i)
    out_path = os.path.join("..","temp","video_detection.mp4")
    pw_utils.process_video(source_path=video, target_path=out_path,
                           callback=cb, target_fps=int(target_fps), codec=codec)
    return out_path

with gr.Blocks() as demo:
    gr.Markdown("# Pytorch-Wildlife Demo.")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["None","MegaDetectorV5","MegaDetectorV6","HerdNet General","HerdNet Ennedi"],
            label="Detection model", value="None"
        )
        det_version = gr.Dropdown(
            ["None"], label="Model version", value="None"
        )

    with gr.Column():
        clf_drop = gr.Dropdown(
            ["None","AI4GOpossum","AI4GAmazonRainforest","AI4GSnapshotSerengeti","CustomWeights","WildlifeModel","SpeciesNet"],
            label="Classification model", interactive=True, visible=False, value="None"
        )
        custom_weights_path = gr.Textbox(
            label="Weights / Model Path",
            visible=False,
            placeholder="./weights/my_weight.pt, wildlife_model_badger_otter.pth, or kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
        )
        custom_weights_class = gr.Textbox(label="Custom Weights Class", visible=False, placeholder="{1:'ocelot',2:'cow',3:'bear'}")
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")

    def update_ui_elements(det_model):
        if det_model=="MegaDetectorV6":
            return (
                gr.Dropdown(
                    choices=["MDV6-yolov9-c","MDV6-yolov9-e","MDV6-yolov10-c","MDV6-yolov10-e","MDV6-rtdetr-c"],
                    label="Model version", value="MDV6-yolov9-e"
                ),
                gr.update(visible=True)
            )
        elif det_model=="MegaDetectorV5":
            return (
                gr.Dropdown(choices=["a","b"], label="Model version", value="a"),
                gr.update(visible=True)
            )
        else:
            return (
                gr.Dropdown(choices=["None"], label="Model version", value="None"),
                gr.update(value="None", visible=False)
            )

    det_drop.change(update_ui_elements, det_drop, [det_version, clf_drop])

    def toggle_textboxes(model):
        if model=="CustomWeights":
            return gr.update(visible=True), gr.update(visible=True)
        if model in {"WildlifeModel", "SpeciesNet"}:
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=False)

    clf_drop.change(toggle_textboxes, clf_drop, [custom_weights_path, custom_weights_class])

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                sgl_conf_sl_clf = gr.Slider(0,1,label="Classification Confidence Threshold",value=0.7)
            sgl_out = gr.Image()
        sgl_but = gr.Button("Detect Animals!")

    with gr.Tab("Folder Separation"):
        with gr.Row():
            with gr.Column():
                inp_path = gr.Textbox(label="Input path", placeholder="./data/")
                out_path = gr.Textbox(label="Output path", placeholder="./output/")
                bth_conf_fs = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                process_btn = gr.Button("Process Files")
                bth_out2 = gr.File(label="Detection Results JSON.")
                process_but = gr.Button("Separate files")
            process_btn.click(batch_path_detection, inputs=[inp_path,bth_conf_fs], outputs=bth_out2)
            process_but.click(pw_utils.detection_folder_separation,
                              inputs=[bth_out2,inp_path,out_path,bth_conf_fs],
                              outputs=out_path)

    with gr.Tab("Batch Image Process"):
        with gr.Row():
            with gr.Column():
                bth_in = gr.File(label="Upload zip file.")
                chck_timelapse = gr.Checkbox(label="Generate timelapse JSON", visible=False)
                bth_conf_sl = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
            bth_out = gr.File(label="Detection Results JSON.")
        bth_but = gr.Button("Detect Animals!")
        bth_but.click(batch_detection, inputs=[bth_in,chck_timelapse,bth_conf_sl], outputs=bth_out)

    with gr.Tab("Single Video Process"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video()
                vid_conf_sl_det = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                vid_conf_sl_clf = gr.Slider(0,1,label="Classification Confidence Threshold",value=0.7)
                vid_fr = gr.Dropdown([5,10,30],label="Output video framerate",value=30)
                vid_enc = gr.Dropdown(["mp4v","avc1"],label="Video encoder",value="mp4v")
            vid_out = gr.Video()
        vid_but = gr.Button("Detect Animals!")
        vid_but.click(video_detection,
                      inputs=[vid_in,vid_conf_sl_det,vid_conf_sl_clf,vid_fr,vid_enc],
                      outputs=vid_out)

    # Show timelapse checkbox only when detection model is not HerdNet
    det_drop.change(
        lambda m: gr.update(visible=True) if "HerdNet" not in m else gr.update(visible=False),
        det_drop, [chck_timelapse]
    )

    load_but.click(
        load_models,
        inputs=[det_drop, det_version, clf_drop, custom_weights_path, custom_weights_class],
        outputs=load_out
    )
    sgl_but.click(
        single_image_detection,
        inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf],
        outputs=sgl_out
    )

# Launch without share (no frpc download)
if __name__ == "__main__":
    demo.launch(share=False)
