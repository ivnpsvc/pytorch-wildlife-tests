from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")

photos = [
    "images/coyote_day.jpg",
    "images/deer_night.jpg",
    "images/raccoon_partial_night.jpg",
    "images/empty_fence_day.jpg",
    "images/empty_hillside_day.jpg",
]

for photo in photos:
    result = model.single_image_detection(photo)
    print(photo, result["labels"])