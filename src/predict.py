import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "resnet18_final_with_metadata.pth"
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

class_mapping = checkpoint["class_mapping"]
threshold = checkpoint["selected_threshold"]

# Convert class ids to class names
class_names = [
    class_mapping[i]
    for i in range(len(class_mapping))
]


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# --------------------------------------------------
# Inference transform
# --------------------------------------------------

inference_transform = transforms.Compose([
    transforms.Resize(
        tuple(checkpoint["transforms"]["resize"])
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["transforms"]["normalize"]["mean"],
        std=checkpoint["transforms"]["normalize"]["std"]
    )
])


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = inference_transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_index = probabilities.argmax().item()
    confidence = probabilities[predicted_index].item()

    predicted_class = class_names[predicted_index]

    probability_dict = {
        class_names[i]: round(
            probabilities[i].item(),
            6
        )
        for i in range(len(class_names))
    }

    result = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "probabilities": probability_dict,
        "needs_review": confidence < threshold
    }

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Traffic vehicle image prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    result = predict_image(args.image)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()