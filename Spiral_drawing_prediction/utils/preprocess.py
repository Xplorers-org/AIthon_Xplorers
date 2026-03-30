from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = cv2.imread(img_path)

    if image is None:
        raise ValueError(f"Image not found: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil)

    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)

    image_np = np.expand_dims(image_np, axis=0)

    return image_np