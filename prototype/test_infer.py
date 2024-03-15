# Paths
trt_file = "weights/fcn-resnet50-12-fp16.engine"
input_file = "demo/airforce_one.jpg"

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from PIL import Image

ctx = pycuda.autoinit.context
ctx.push()

def preprocess(image):
    # Mean normalization
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

# Load Engine
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(trt_file, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Reading input image from file {}".format(input_file))
with Image.open(input_file) as img:
    input_image = preprocess(img)
    image_width = img.width
    image_height = img.height
    print ("Input Image Shape: ", input_image.shape)

with engine.create_execution_context() as context:
    # Set input shape based on image dimensions for inference
    context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
    # Allocate host and device buffers
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        print("Binding IDX: ", binding_idx)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            print("Input Shape: ", context.get_binding_shape(binding_idx))
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes)
            bindings.append(int(input_memory))
        else:
            print("Output Shape: ", context.get_binding_shape(binding_idx))
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # Run inference
    # context.execute_async(bindings=bindings, stream_handle=stream.handle)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer prediction output from the GPU.
    # print("Copying Output")
    # cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    # Synchronize the stream
    stream.synchronize()
    # context.pop()

ctx.pop()