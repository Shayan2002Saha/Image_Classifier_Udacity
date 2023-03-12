# Importing Python modules
import torch
from torch import nn
from torchvision import models

def create_model(arch, hidden_units, output_units):
    """
    Creates the model using the given pretrained model, freezes the parameters and replaces the       classifier with a new classifier
    
    Parameters: Name of pretrained model, List of hidden units, Number of output units
    
    Returns: Newly created model
    """
    # Dictionary defining the models
    archs = {'vgg13': models.vgg13(weights='DEFAULT'), 
             'densenet121': models.densenet121(weights='DEFAULT'),
             'resnet18': models.resnet18(weights='DEFAULT')}
    
    # Loading the model
    model = archs[arch]
    
    # Freezing the parameters
    for param in model.parameters():
        param.requires_grad=False
    
    # Getting the number of input units based on the model
    input_units = None
    if arch == 'vgg13':   
        input_units = model.classifier[0].in_features
    elif arch == 'densenet121':
        input_units = model.classifier.in_features
    else:
        input_units = model.fc.in_features
    
    # Defining the hidden layers
    hidden_layers = []
    if len(hidden_units) > 1:
        for h1, h2 in zip(hidden_units[:-1], hidden_units[1:]):
            hidden_layers.extend([nn.Linear(h1,h2), nn.ReLU(), nn.Dropout(0.2)])
    
    # Defining the new classifier
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units[0]),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              *hidden_layers,
                              nn.Linear(hidden_units[-1], output_units),
                              nn.LogSoftmax(dim=1))
    
    # Replacing the model classifier with the new classifier
    if arch == 'resnet18':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model

def train_model(model, dataloaders, epochs, device, criterion, optimizer):
    '''
    Function to train the model
    
    Parameters:
    model: The model to be trained
    dataloaders: The dataloaders containg training, validating and testing data
    epochs: Number of epochs
    device: The device on which model will be trained
    criterion: The Loss function used (NLLLoss)
    optimizer: The optimizer used for gradient descent (Adam)
    
    Return: None
    '''
    steps = 0
    running_trainloss = 0
    print_every = 10
    
    # Moving model to available device
    model.to(device)
    model.train()
    # Start of training process
    for epoch in range(epochs):
        for images, labels in dataloaders['train']:
            steps +=1 
            # Setting gradients to zero
            optimizer.zero_grad()
            # Moving images and labels to the given device
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_trainloss+=loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                running_validloss = 0
                
                # Checking the accuracy using validation set
                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        running_validloss += loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

                print(f"Epochs: {epoch+1}/{epochs}..."
                      f"Current trainloss: {running_trainloss/print_every:.3f}..."
                      f"Current validation loss: {running_validloss/len(dataloaders['valid']):.3f}..."
                      f"Current accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_trainloss =0
                model.train()
                
def test_model(model, dataloaders, criterion, device):
    '''
    Function to test the model on test set
    
    Parameters:
    model: The trained model
    dataloaders: The dataloader containg the test data
    criterion: The loss function to calculate the loss
    
    Return: None
    '''
    
    model.eval()
    accuracy = 0
    testloss = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            testloss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equal = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

    print(f"Testloss: {testloss/len(dataloaders['test']):.3f}..."
          f"Accuracy: {accuracy/len(dataloaders['test']):.3f}")