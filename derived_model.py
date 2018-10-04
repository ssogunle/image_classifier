### Imports ###

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable

# Helper files
from ffclassifier import FFClassifier

class DerivedModel:
    ''' Defines a model with custom functionalities '''

    def __init__(self):
        #### Parameters ####
        self.base_model = None # Pretrained model 
        self.base_model_arch = None # The name of the base model
        self.input_size = 0 #Input size for the classifier
        self.class_to_idx = None # Data Class-to-Index Mapping
        self.learn_rate = None # Network's learning rate
        self.device = None # CPU or GPU
        self.epochs = None # Training epochs
        self.optimizer = None  # Model optimizer
        self.criterion = None  # Error function
        
    def load_base_model(self, arch_name):
        ''' Loads a base model 
            Returns name and object of the base model, input_size
        '''
        self.base_model_arch, self.base_model, self.input_size = self.get_pretrained_model(arch_name)
    
    def get_pretrained_model(self, arch_name):
        ''' Loads a pre-trained model '''

        input_size = 0
        model = None

        # Loads a Pre-trained network
        if arch_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = model.classifier[0].in_features
        elif arch_name == 'densenet161':
            model = models.densenet161(pretrained=True)
            input_size = model.classifier.in_features
        elif arch_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = model.classifier[1].in_features
                  
        if model == None:
            raise Exception('Specified model is not supported')
            exit()
        else:
            # Freezes parameters for the selected model i.e., no backproping through them
            for param in model.parameters():
                param.requires_grad = False
        
        return arch_name, model, input_size
    

    def assign_new_classifier(self, output_size, hidden_layer_sizes, image_dataset, drop_p=0.5):
        ''' Creates new classifier for the pre-trained model '''
        
        # Gets class-to-index mapping from data set 
        self.class_to_idx = image_dataset.class_to_idx
        
        # Instantiate FFClassifier
        # Assigns pre-trained network's input size to new classifer input size
        
        ffclassifier = FFClassifier(self.input_size, output_size, hidden_layer_sizes, drop_p)
        
        # Assigns newly created classifier to the pre-trained model
        self.base_model.classifier = ffclassifier
        
        
    def assign_optimizer(self, learn_rate=0.0001, optimizer=None):
        ''' 
            Function called after classifier is created.
            Assigns a model optimizer.
            Default:  Adam
        '''
        
        self.learn_rate = learn_rate
        
        if optimizer == None:
            self.optimizer = optim.Adam(self.base_model.classifier.parameters(), lr=learn_rate) 
        else:
            self.optimizer = optimizer

            
    def deep_learn(self, dataloader, epochs=3, print_every=40, use_gpu=False, criterion=None):
        ''' Performs deep learning on training data set and track progress with validation data set '''
    
        # Epochs
        self.epochs = epochs 
        
        # Error Function: Default(Non-Linear Log Loss)
        self.criterion = nn.NLLLoss() if criterion == None else criterion
        

        # Determine the device to use
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        if self.device == torch.device('cuda'):
            print("\n************************************\n\tTraining with GPU\n************************************")
        else:
            print("\n************************************\n\tTraining with CPU\n************************************")
        device = self.device #for readability
        
        
        # Pass model to appropriate device
        self.base_model.to(device)

        # Initialize batch instance counter
        steps = 0
        
        for e in range(epochs):
            
            # Initialize loss calculation per epoch
            running_loss = 0

            for ii, (inputs, labels) in enumerate(dataloader['train']):
                
                # Incremement batch count
                steps += 1
 
                # Convert tensors to autograd tensors
                inputs = Variable(inputs)
                labels = Variable(labels)
            
                # Pass tensors to available device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradient buffers
                self.optimizer.zero_grad()

                ### Forward and backward passes ###
                outputs = self.base_model.forward(inputs)  # Feed-forward
                loss = self.criterion(outputs, labels) # Calculate error
                loss.backward()  # Backward pass: Backprop 
                self.optimizer.step() # Update weights

                running_loss += loss.item() # Record loss

                if steps % print_every == 0:
                    
                    # Model will be in inference mode and dropouts will be turned off
                    test_loss, accuracy = self.validate(dataloader['valid'])

                    print("\nEpoch: {}/{}.. ".format(e+1, epochs),
                          "\t\tTraining Loss: {:.3f}.. ".format(running_loss/print_every),
                          "\nValidation Loss: {:.3f}.. ".format(test_loss),
                          "\tValidation Accuracy: {:.2f} %".format(accuracy))
                    
                    # Initialize loss calculation per batch
                    running_loss = 0

                    # Make sure training is back on: dropouts turned on
                    self.base_model.train()

                    
    def validate(self, testloader):
        ''' Tracks network's prediction accuracy as it learns:
                
            Parameter(s):
                testloader: image_dataset to be used
        '''
        test_loss = 0
        accuracy = 0
        device = self.device
        
        # Set model to eval mode for inferencing: Turn off dropouts
        self.base_model.eval()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            for images, labels in testloader:
                # Convert tensors to autograd tensors
                images = Variable(images)
                labels = Variable(labels)

                images, labels = images.to(device), labels.to(device)

                output = self.base_model.forward(images)
                test_loss += self.criterion(output, labels).item()
                
                # Take the exponential of log-softmax output to get corresponding probabilities
                ps = torch.exp(output)

                # Find class with the highst probability and compare with true label
                equality = (labels.data == ps.max(dim=1)[1])
                
                # Compute Accuracy by taking the mean 
                accuracy += equality.type(torch.FloatTensor).mean()

        # Normalize
        data_size = len(testloader)
        test_loss = test_loss/data_size 
        accuracy = 100*accuracy/data_size # Convert to percentage
        
        return test_loss, accuracy 
