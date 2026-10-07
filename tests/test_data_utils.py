import numpy as np
import pytest
import torch
from PIL import Image
from PytorchWildlife.data import datasets, transforms

PADDING_GRAY = 114 / 255


@pytest.mark.parametrize(
    "filename",
    [
        "photo.jpg",
        "PHOTO.JPG",
        "photo.Jpeg",
        "photo.png",
        "photo.tif",
        "photo.tiff",
        "photo.bmp",
        "photo.webp",
    ],
)
def test_image_extensions_are_accepted(filename):
    assert datasets.is_image_file(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "notes.txt",
        "photo.jpg.txt",
        "jpg",
        "animation.gif",
        "photo.heic",
        "photo.raw",
    ],
)
def test_other_files_are_not_images(filename):
    assert not datasets.is_image_file(filename)


@pytest.mark.parametrize("height, width", [(1494, 2048), (2048, 1494), (100, 100)])
def test_letterbox_output_is_a_square_of_the_target_size(height, width):
    image = torch.rand(3, height, width)
    output = transforms.letterbox(image, new_shape=1280)
    assert tuple(output.shape) == (3, 1280, 1280)


def test_letterbox_keeps_the_aspect_ratio():
    # A white landscape photo, 2048 wide and 1494 high, scaled to 1280 wide,
    # should be round(1494 * 1280 / 2048) = 934 rows high. The rest is padding.
    image = torch.ones(3, 1494, 2048)
    output = transforms.letterbox(image, new_shape=1280)
    white_rows = (output[0] == 1.0).all(dim=1).sum().item()
    assert white_rows == pytest.approx(934, abs=1)


def test_letterbox_pads_with_gray():
    image = torch.ones(3, 1494, 2048)
    output = transforms.letterbox(image, new_shape=1280)
    top_left_corner = output[:, 0, 0]
    assert top_left_corner.tolist() == pytest.approx([PADDING_GRAY] * 3)


def test_letterbox_accepts_a_pil_image(images_dir):
    photo = Image.open(images_dir / "coyote_day.jpg")
    output = transforms.letterbox(photo, new_shape=640)
    assert tuple(output.shape) == (3, 640, 640)


def test_megadetector_transform_gives_scaled_square_tensor(images_dir):
    photo = np.array(Image.open(images_dir / "coyote_day.jpg").convert("RGB"))
    transform = transforms.MegaDetector_v5_Transform(target_size=1280)
    output = transform(photo)
    assert tuple(output.shape) == (3, 1280, 1280)
    assert output.min() >= 0
    assert output.max() <= 1


def test_classification_transform_gives_224_square(images_dir):
    photo = Image.open(images_dir / "coyote_day.jpg").convert("RGB")
    transform = transforms.Classification_Inference_Transform(target_size=224)
    assert tuple(transform(photo).shape) == (3, 224, 224)
