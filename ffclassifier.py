### Imports ###

# PyTorch Libraries
import torch.nn as nn
import torch.nn.functional as F

class FFClassifier(nn.Module):
    ''' Defines Feedforward Classifier'''
    def __init__(self, input_size, output_size, hidden_layer_sizes, drop_p=0.5):
        
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers, self.output = self.compute_connected_layers()
        
        ### Dropout probability
        self.dropout_p = drop_p
        
        # Prevent overfitting with Dropout
        self.dropout = nn.Dropout(p=drop_p)   
        
      
    def forward(self, inputs):
        ''' Forward pass through the network '''
        
        # Feed-Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            inputs = F.relu(linear(inputs))
            inputs = self.dropout(inputs)
    
        # Feed-Forward toward output layer
        inputs = self.output(inputs)

        # Apply Log-Softmax
        optimized_output = F.log_softmax(inputs, dim=1)
        
        return optimized_output
    
    def compute_connected_layers(self):
        ''' Creates a deep network for the classifier '''

        # Create ModuleList and input-to-first-hidden layer connection
        hidden_layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_layer_sizes[0])])

        # Create inter-hidden-layer connections
        layer_sizes = zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])
        # Add hidden layers to the ModuleList
        hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])   
        
        # Create a last-hidden-to-output layer connection
        output = nn.Linear(self.hidden_layer_sizes[-1], self.output_size)

        return hidden_layers, output
