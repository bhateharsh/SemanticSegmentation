#!/usr/bin/python3
"""
Test out fcn-restnet50.onnx taken from ONNX model zoo
https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/fcn 
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

from PIL import Image
import numpy as np
import torch
from torchvision import transforms, models

from matplotlib.pyplot import imshow


from onnx import numpy_helper
import os
import onnxruntime as rt

from matplotlib.colors import hsv_to_rgb
from matplotlib import pyplot as plt

import cv2


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image = Image.open("demo/000000025205.jpg")
orig_tensor = np.asarray(input_image)
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.detach().cpu().numpy()

imshow(orig_tensor)
plt.show()
sess = rt.InferenceSession("weights/fcn-resnet50-12-int8.onnx", providers=['CUDAExecutionProvider'])

outputs = sess.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name = sess.get_inputs()[0].name

detections = sess.run(output_names, {input_name: input_tensor})
print("Output shape:", list(map(lambda detection: detection.shape, detections)))
output, aux = detections

classes = [line.rstrip('\n') for line in open('dependencies/voc_classes.txt')]
num_classes = len(classes)

def get_palette():
    # prepare and return palette
    palette = [0] * num_classes * 3
    
    for hue in range(num_classes):
        if hue == 0: # Background color
            colors = (0, 0, 0)
        else:
            colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))
            
        for i in range(3):
            palette[hue * 3 + i] = int(colors[i] * 255)
            
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P', colors=num_classes)
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def visualize_output(image, output):
    assert(image.shape[0] == output.shape[1] and \
           image.shape[1] == output.shape[2]) # Same height and width
    assert(output.shape[0] == num_classes)
    
    # get classification labels
    raw_labels = np.argmax(output, axis=0).astype(np.uint8)

    # comput confidence score
    confidence = float(np.max(output, axis=0).mean())

    # generate segmented image
    result_img = colorize(raw_labels)
    
    # generate blended image
    blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img, 0.5, 0)
    
    result_img = Image.fromarray(result_img)
    blended_img = Image.fromarray(blended_img)

    return confidence, result_img, blended_img, raw_labels

conf, result_img, blended_img, _ = visualize_output(orig_tensor, output[0])
print(conf)
imshow(np.asarray(result_img))
plt.show()

imshow(np.asarray(blended_img))
plt.show()

aux_conf, aux_result, aux_blended, _ = visualize_output(orig_tensor, aux[0])
print(aux_conf)
imshow(np.asarray(aux_result))
plt.show()

imshow(np.asarray(aux_blended))
plt.show()