import torch.onnx
import torchvision
from colors import bcolors
    
BATCH_SIZE = 1
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)

model = torchvision.models.resnet50(pretrained=True, progress=False).eval()
torch.onnx.export(model, dummy_input, "./pretrained_model/Pytorch/resnet50.onnx",
                  verbose=False, input_names=['input'], output_names=['output'])
torch.save(model, "./pretrained_model/pytorch/resnet50.pt")
print(bcolors.OKGREEN+"resnet50 Done")

model = torchvision.models.resnet101(pretrained=True, progress=False).eval()
torch.onnx.export(model, dummy_input, "./pretrained_model/Pytorch/resnet101.onnx",
                  verbose=False, input_names=['input'], output_names=['output'])
torch.save(model, "./pretrained_model/pytorch/resnet101.pt")
print(bcolors.OKGREEN+"resnet101 Done")

model = torchvision.models.googlenet(pretrained=True, progress=False).eval()
torch.onnx.export(model, dummy_input, "./pretrained_model/Pytorch/googlenet.onnx",
                  verbose=False, input_names=['input'], output_names=['output'])
torch.save(model, "./pretrained_model/pytorch/googlenet.pt")
print(bcolors.OKGREEN+"googlenet Done")
