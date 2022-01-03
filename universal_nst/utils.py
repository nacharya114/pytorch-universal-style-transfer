import torch
import torch.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_imsize():
    return 512 if torch.cuda.is_available() else 256

def preprocess(imsize=None):
    if not imsize:
        imsize = get_imsize()
    return transforms.Compose([
          transforms.Resize(imsize),
          transforms.CenterCrop(imsize),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

def image_to_tensor(filename):
    input_image = Image.open(filename)
    imsize = 512 if torch.cuda.is_available() else 256
    input_tensor = preprocess(imsize)(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch.to(device)

def imsave(tensor, filename):
    image = tensor.cpu().clone()
    inv_normalize = transforms.Normalize(
      mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
      std=[1/0.229, 1/0.224, 1/0.225]
    ) 
    image = inv_normalize(image)

    save_image(image, filename)