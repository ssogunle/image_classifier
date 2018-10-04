### Imports ###
import os.path
from PIL import Image

# Numpy
import numpy as np

# Pytorch
import torch
from torchvision import datasets, transforms

def check_file_folder(specified_path, is_folder):
    ''' Verifies file and folder validity '''
    abs_path = None 
    try:
        abs_path = os.path.abspath(specified_path) # Get absolute path
        if is_folder and os.path.isdir(abs_path)!=True:
            raise Exception("Folder with path '{}' not found".format(abs_path))
        elif not is_folder and os.path.isfile(abs_path)!=True:
            raise Exception("File with path '{}' not found".format(abs_path))
    except Exception as e:
        print(e)
        exit() #Halt program

    return abs_path

def load_data_set(data_dir):
    # Transforms for all data sets
    t_to_tensor = transforms.ToTensor()
    t_normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )

    # Training set transforms
    t_random_rotation = transforms.RandomRotation(75) 
    t_random_resize_crop = transforms.RandomResizedCrop(224) 
    t_random_h_flip  = transforms.RandomHorizontalFlip()

    # Validation and Testing set transforms
    t_resize = transforms.Resize(256)
    t_center_crop = transforms.CenterCrop(224)

    # Defining transforms for the training, validation, and testing sets
    data_transforms = {
            'train': transforms.Compose([
                t_random_rotation, 
                t_random_resize_crop,
                t_random_h_flip,
                t_to_tensor, 
                t_normalize
            ]),
            'valid': transforms.Compose([
                t_resize,
                t_center_crop,
                t_to_tensor, 
                t_normalize
            ]),
            'test': transforms.Compose([
                t_resize,
                t_center_crop,
                t_to_tensor, 
                t_normalize
            ]),
    }

    # Data Loading: Defining datasets
    image_datasets = {
                data: datasets.ImageFolder(data_dir+'/'+data, transform=data_transforms[data]) 
                for data in data_transforms
    }

    # Data Batching: Defining dataloaders for image_datasets
    dataloaders = {
            data: torch.utils.data.DataLoader(image_datasets[data], batch_size=64, shuffle=True)
            for data in image_datasets
    }

    return dataloaders, image_datasets 
    
    
def process_image(image_file):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Resize Image
    size = 256, 256 
    pil_image = Image.open(str(image_file)) # Load Image

    pil_image.thumbnail(size, Image.ANTIALIAS) #Image.ANTIALIAS to keep aspect ratio 
  
    # Center-crop Image
    pil_image = centered_crop(pil_image, 224, 224)
    
    # Possible color channels
    color_channels = 255
    
    # Convert Image (with channels) to Numpy Array (with values between 0 and 1)
    np_image = np.array(pil_image) / color_channels
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image_array = (np_image - mean) / std
    
    # Reorder dimensions
    np_image_array = np_image_array.transpose((2, 0, 1))

    return np_image_array

def centered_crop(pil_image, new_height, new_width):
    '''
    Code adapted from: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    Accessed: 30.09.2018
    ''' 
    width, height = pil_image.size  
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    centered_cropped_image = pil_image.crop((left, top, right, bottom))  # Image.crop rectangular box
    
    return centered_cropped_image