### Imports ###

# Python Base Libraries
from collections import OrderedDict
import json
import argparse
import os.path

# Pytorch Core Libraries
import torch
import torchvision
from torchvision import datasets, transforms, models

# Pytorch Libraries for Deep Learning 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Pytorch Library for Loss Function and Optimization
import torch.optim as optim

# Helper files
from ffclassifier import FFClassifier
from derived_model import DerivedModel
import utils


def save_checkpoint(model, chkpt_pth, data_dir):
    '''Save the checkpoint '''

    done = False
    model_checkpoint = {

        # Classifier-specific parameters
        'arch': model.base_model_arch,
        'output_size': model.base_model.classifier.output_size,
        'hidden_layers': model.base_model.classifier.hidden_layer_sizes,
        'dropout_p': model.base_model.classifier.dropout_p, 

        # Model-specific parameters    
        'data_dir': data_dir,
        'epochs': model.epochs,
        'learning_rate': model.learn_rate,
        'optimizer' : model.optimizer.state_dict(),
        'state_dict': model.base_model.state_dict() 

    }

    torch.save(model_checkpoint, chkpt_pth)
    done = True

    return done

def get_input_args():
    
    # Create Argument Parser object named parser
    parser = argparse.ArgumentParser()
    

    # Add argument for folder path for checkpoint
    parser.add_argument('--save_dir', type = str, default = 'model_ckpt.pth', 
                    help = 'Path to save checkpoint file') 
    
    # Add argument for CNN model architecture
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'CNN model architecture')
    
    # Add argument for hyperparameter: learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.0001, 
                    help = 'Model\'s learning rate')
    
    # Add argument for hyperparameter: training epochs
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'Training epochs number') 
    
    # Add argument for hyperparameter: hidden-units
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[1024, 512, 256],
                    help = 'Hidden layer sizes for the network') 
    
    # Add argument for parameter: gpu
    parser.add_argument('--gpu',  action='store_true',
                    help = 'Uses GPU for training the model') 

    # Add argument for parameter: dropout probability
    parser.add_argument('--dropout', type = float, default=0.33,
                    help = 'Specifies the dropout probability for training the model')
    
    # Add argument for parameter: displaying stats every n data point count
    parser.add_argument('--print_every', type = int, default=40,
                    help = 'Displays stats every n data point count ')
       
   
    in_args, extras = parser.parse_known_args()
    
    return in_args, extras

def main():

    ''' Pre-process inputs '''
    
    # Define base argument(s)
    _output_size = 102 # Model's output size
    data_dir = os.path.abspath('flowers') # Main data directory

    # Parse CLI input argument(s)
    in_args, extras  = get_input_args()
   
    data_dir = utils.check_file_folder(extras[0], True) # Extract data set path
        
    # Parameters with default argparse values
    _checkpoint_path = in_args.save_dir # checkpoint file path
    
    checkpoint_path = os.path.abspath(_checkpoint_path)
    
    arch_name = in_args.arch # model architecture
    _learn_rate = in_args.learning_rate # learning rate
    _epochs = in_args.epochs # training epochs
    hidden_units = in_args.hidden_units # hidden layer sizes
    _use_gpu = in_args.gpu # gpu flag
    _dropout_p =  in_args.dropout # dropout probability
    _print_every = in_args.print_every # Print stats every batch instance count
 
    # Load data set
    data_dir_name = str(data_dir.split('/')[-1])
    _dataloaders, _image_datasets = utils.load_data_set(data_dir_name)
    
    ''' Build and train the network '''
    # Instantiate Derived Model
    dmodel = DerivedModel()
    dmodel.load_base_model(arch_name)
    dmodel.assign_new_classifier(_output_size, hidden_units, _image_datasets['train'], _dropout_p)
    dmodel.assign_optimizer(_learn_rate)
   
    # Train model: Do Deep Learning
    dmodel.deep_learn(_dataloaders, epochs=_epochs, print_every=_print_every, use_gpu=_use_gpu)
       

    ''' Save the model state '''
    # Save checkpoint
    if save_checkpoint(dmodel, checkpoint_path, data_dir_name):
        print("\nCheckpoint saved in {}".format(checkpoint_path))
        
if __name__ == '__main__':
    main()
  