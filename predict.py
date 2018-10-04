### Imports ###

# Python modules
import argparse
import os.path
import json

# Numpy
import numpy as np

# Pytorch libraries
import torch
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable

# Helper classes
from ffclassifier import FFClassifier
from derived_model import DerivedModel
import utils

def load_checkpoint(checkpoint_path, use_gpu):
    ''' Loads a checkpoint and rebuilds the model '''

    if use_gpu and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
    # Retrieve variables for classifier
    data_dir_name = checkpoint['data_dir']
    arch_name = checkpoint['arch']
    out_size = checkpoint['output_size']
    hidden_layer_sizes = checkpoint['hidden_layers']
    dropout_p = checkpoint['dropout_p']

    # Retrieve variables for model
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    
    _, _image_datasets = utils.load_data_set(data_dir_name)
    
    # Re-build model
    model = DerivedModel() 
    model.load_base_model(arch_name)
    model.assign_new_classifier(out_size, hidden_layer_sizes, _image_datasets['train'], dropout_p) 
    model.assign_optimizer(learning_rate) 
    model.base_model.load_state_dict(checkpoint['state_dict']) 
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    print("\nModel from '{}' has been successfully loaded".format(checkpoint_path))
    return model  

def predict(image_path, model, k=1, use_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''

    #Pre-process Image 
    np_array = utils.process_image(image_path) # Load & Format PIL Image to Numpy Image Array
    tensor_input = torch.from_numpy(np_array) # Convert image_array to torch tensor input
    
    # Determine the device to use
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print("\n************************************\n\tPredicting with GPU\n************************************")
    else:
        print("\n************************************\n\tPredicting with CPU\n************************************")

    # Turn off dropouts
    model.base_model.eval()
    
    # Pass model to the appropriate device
    model.base_model.to(device)
    
    with torch.no_grad():
        # Autograd Tensor
        m_input = Variable(tensor_input)

        # Pass input object to device
        m_input = m_input.to(device)

        # Add one more dimension
        m_input = m_input.unsqueeze(0) 
        
        # Feed input into model
        m_output = model.base_model.forward(m_input.float())
    
        # Top K probabilities
        outcome = torch.exp(m_output).data.topk(k)
    
    # For to_numpy conversion
    if use_gpu:
        probs = outcome[0].cpu()
        classes = outcome[1].cpu()
    else:
        probs = outcome[0]
        classes = outcome[1]
    
    # Invert class_to_idx dictionary
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    # Retrieve predicted class indices 
    pred_classes = [idx_to_class[label] for label in classes.numpy()[0]]
    
    return probs.numpy()[0], pred_classes

def get_input_args():
    
    # Create Argument Parser object named parser
    parser = argparse.ArgumentParser()
    
    # Add argument for parameter: top-k
    parser.add_argument('--top_k', type = int, default = 1, 
                    help = 'Prints top k probabilities for an image prediction') 
    
    # Add argument for parameter: class-name-file
    parser.add_argument('--category_names', type = str, default='cat_to_name.json',
                    help = 'Loads JSON file that maps class values to category names') 
    
    # Add argument for parameter: gpu
    parser.add_argument('--gpu',  action='store_true',
                    help = 'Uses GPU for making predictions') 
    
    in_args, extras = parser.parse_known_args()
    
    return in_args, extras


def main():
    
    # Default values
    image_path = None
    category_names_file = None
    checkpoint_pth = None
    
    # Parse input arguments
    in_args, extras  = get_input_args()
    
    try:
        image_path = utils.check_file_folder(extras[0], False) # Extract input image path
        checkpoint_pth = utils.check_file_folder(extras[1], False) # Extract checkpoint path
        category_names_file = utils.check_file_folder(os.path.abspath(in_args.category_names), False) # Extract category name file
    except Exception as e:
        print("\nError:",e,"\nAt least one of the expected arguments was not valid or specified")
        exit() # Halt program
    
    # Extract argparse argument values
    k = in_args.top_k # Top k predictions for the input image
    _use_gpu = in_args.gpu # Use GPU to make predictions 
    
    # Read the checkpoint 
    dmodel = load_checkpoint(checkpoint_pth, _use_gpu)

    # Load JSON file with associated class names
    with open(category_names_file , 'r') as f:
        cat_to_name = json.load(f)
        
    ''' Print actual test image category name '''
    image_cat = str(image_path.split('/')[-2])
    flower_class = cat_to_name[image_cat]
    print("\nOriginal flower class: ", flower_class.title())

    # Ensure lowest value for k
    if k<1:
        k = 1

    # Read and predict test image
    probs, classes = predict(image_path, dmodel, k, _use_gpu)
    
    ''' Print most likely image class and associated probability '''
    # Retrieve the possition of the highest prediction
    max_pred_pos = np.argmax(probs)
    # Retrieve highest prediction
    max_pred = probs[max_pred_pos]
    # Retrieve corresponding class pair for chosen prediction
    max_pred_class = classes[max_pred_pos]
    
    print("\nMost likely flower class: {}, Probability: {:.3f}".format(cat_to_name[max_pred_class].title(),max_pred))
    
    ''' Print top k classes with associated probabilities '''
    if k>1:
        print("\nTop {} classes with their probabilities:\n".format(k))
        for cl, prob in zip(classes,probs):
            print("\t{}, {:.3f}".format(cat_to_name[cl].title(),prob))
    
if __name__ == '__main__':
    main()
