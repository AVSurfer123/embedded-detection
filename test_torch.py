from torchvision.io.image import read_image, write_png
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, resize
import time
import cv2

img = read_image("Cars-Rover.jpeg")
print(img.shape)
scale = 320 / img.shape[2]
rescaled = resize(img, (int(img.shape[1] * scale), int(img.shape[2] * scale)), antialias=True)

# Step 1: Initialize model with the best available weights
# weights = SSDLite320_MobileNet_V3_Large_Weights.
model = ssdlite320_mobilenet_v3_large(pretrained=True, score_thresh=0.5)
model.eval()

# Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
# batch = [preprocess(img)]
batch = [img / 255.0]

# Step 4: Use the model and visualize the prediction
start = time.time()
prediction = model(batch)[0]
print(prediction)
print(f"Prediction took {time.time() - start} seconds")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in prediction["labels"]]
box = draw_bounding_boxes(rescaled, boxes=prediction["boxes"] * scale,
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)

write_png(box, 'detections.png')
