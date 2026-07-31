import argparse
from PIL import Image

from wildlife_classifier import WildlifeClassificationAdapter

DEVICE = "cpu"
MODEL_PATH = "wildlife_model.pth"


def load_model():
    return WildlifeClassificationAdapter(weights_path=MODEL_PATH, device=DEVICE)


def classify_image(model, image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    return model.single_image_classification(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single image using wildlife_model.pth")
    parser.add_argument("image", help="Path to the image to classify")
    args = parser.parse_args()

    model = load_model()
    result = classify_image(model, args.image)
    print(f"Predicted class id: {result['class_id']}")
    print(f"Predicted label: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
