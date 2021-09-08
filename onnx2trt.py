from onnx import ModelProto
import torch
import torchvision
import tensorrt as trt
import sys
import pycuda.driver as cuda
import pycuda.autoinit
from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)


# use Logger.VERBOSE to output optimization process
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape=[1, 224, 224, 3]):
   """
   create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
       builder.max_workspace_size = (256 << 20)
       """256 MB"""
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_cuda_engine(network)
       return engine


def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)


def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

#    gpu = {
#        "RTX_2060": "0",
#        "GTX_1080_Ti": "1"
#    }


def main():
    try:
        # source pretrained model path
        onnx_path = str(sys.argv[1])+".onnx"
        # engine save path
        trt_path = sys.argv[1]+"-"+sys.argv[3]+".plan"
        gpu_idx = str(sys.argv[2])
        gpu_name = str(sys.argv[3])
    except IndexError:
        print("Index error")
        print("You should input model name as CLI arguments")

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    batch_size = 1
    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size, d0, d1, d2]

    # choose GPU before build engine
    # ref:
    #   https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#faq
    #   https://github.com/NVIDIA/TensorRT/issues/1050
    cudaSetDevice(int(gpu_idx))

    # build engine and start timer to get total build time
    trt_engine = build_engine(onnx_path, shape)

    # save engine as trt_path
    save_engine(trt_engine, trt_path)


if __name__ == "__main__":
    main()
