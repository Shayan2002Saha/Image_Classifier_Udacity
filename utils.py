# Imports Python modules
import argparse
import torch
from torch import nn
from PIL import Image
from torchvision import datasets, transforms, models

def get_train_arguments():
    """
    Parses command line arguments given by the user when executing the training program
    using command line.
    This function uses Python's argparse module to create and define command line arguments.
    If some arguments are not provided by the user, default values are used.
    
    Positional Arguments:
    data_dir: Directory containing the datasets
    
    Optional Arguments:
    --save_dir: Dictionary for saving checkpoint
    --arch: Name of the pretrained model. Choices: ['vgg13', 'densenet121', 'resnet18']. Default: 'densenet121'
    --learning_rate: Learning rate of the model. Default: 0.003
    --hiddent_units: List of hidden units. Default: [512,256]
    --epochs: Number of epochs. Default: 10
    --gpu: Flag for using GPU during training. Default: False
    
    Parameters: None
    
    Return:
    ArgumentParser object containing the arguments
    """
    # Creates the parser as ArgumentParser object
    parser = argparse.ArgumentParser(description="Trains an Image Classifier")
    
    # Creating the arguments as defined above using add_argument() method
    parser.add_argument("data_dir", type = str, help="Provide directory containing datasets")
    parser.add_argument("--save_dir", type = str, help="Provide directory for saving checkpoint")
    parser.add_argument("--arch", type = str, choices=['vgg13', 'densenet121', 'resnet18'], default='densenet121', help="Provide name of pretrained model")
    parser.add_argument("--learning_rate", type = float, default='0.003', help="Provide learning_rate")
    parser.add_argument("--hidden_units", type = int, nargs='+', default=[512,256], help="Provide hidden units")
    parser.add_argument("--epochs", type = int, default=10, help="Provide number of epochs")
    parser.add_argument("--gpu", action='store_true', help="Use if you want to train with GPU")
    
    # Retrieves the user-entered arguments from command line
    args = parser.parse_args()
    
    # Returns the collection of arguments
    return args

def load_datasets(data_dir):
    """
    Loads the train, validate and test datasets with appropriate transforms and defines dataloaders
    
    Parameters: Data directory containing the datasets
    
    Return: image datasets and dataloaders
    """
    # Separating the training, validatind and testing datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining transforms
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])}
    
    # Loading the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                     'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                     'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # Using the image datasets and the transforms, defining the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                  'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
                  'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)}
    
    return image_datasets, dataloaders

def get_predict_arguments():
    """
    Parses command line arguments given by the user when executing the predict program
    using command line.
    This function uses Python's argparse module to create and define command line arguments.
    If some arguments are not provided by the user, default values are used.
    
    Positional Arguments:
    input: Path of the input image
    checkpoint: Path of the model checkpoint
    
    Optional Arguments:
    --top_k: Value of k for top k predictions, Default = 1
    --category_names: Path to the file containing mapping of categories to real names
    --gpu: Flag for using GPU during testing. Default: False
    
    Parameters: None
    
    Return:
    ArgumentParser object containing the arguments
    """
    # Creates the parser as ArgumentParser object
    parser = argparse.ArgumentParser(description="Predicts class of image")
    
    # Creating the arguments as defined above using add_argument() method
    parser.add_argument("input", type = str, help="Provide path of the input image")
    parser.add_argument("checkpoint", type = str, help="Provide path of model checkpoint")
    parser.add_argument("--top_k", type = int, default=1, help="Provide value of k for top_k predictions")
    parser.add_argument("--category_names", type = str, help="Provide path to the file containing mapping of categories to real names")
    parser.add_argument("--gpu", action='store_true', help="Use if you want to predict with GPU")
    
    # Retrieves the user-entered arguments from command line
    args = parser.parse_args()
    
    # Returns the collection of arguments
    return args

def load_checkpoint(path):
    '''
    This function loads a checkpoint and rebuilds the model
    
    Parameter: Path to the checklpoint
    
    Return: The model rebuilt using the checkpoint
    '''
    checkpoint = torch.load(path, map_location='cpu')
    
    # Dictionary defining the models
    archs = {'vgg13': models.vgg13(pretrained=True), 
             'densenet121': models.densenet121(pretrained=True),
             'resnet18': models.resnet18(pretrained=True)}
    
    model = archs[checkpoint['model']]
    
    # Loading the mapping of class to label
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Freezing the parameters
    for params in model.parameters():
        params.requires_grad = False
    
    # Defining the hidden layers
    hidden_units = checkpoint['hidden_layer']
    hidden_layers = []
    if len(hidden_units) > 1:
        for h1, h2 in zip(hidden_units[:-1], hidden_units[1:]):
            hidden_layers.extend([nn.Linear(h1,h2), nn.ReLU(), nn.Dropout(0.2)])
        
    # Defining the new classifier
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], hidden_units[0]),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              *hidden_layers,
                              nn.Linear(hidden_units[-1], checkpoint['output_size']),
                              nn.LogSoftmax(dim=1))
    
    # Replacing the model classifier with the new classifier
    if checkpoint['model'] == 'resnet18':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    # Loading the state of the model parameters
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    
    Parameter: Path of the image
    
    Return: Transformed image converted to a Numpy array
    '''
    
    # Processing a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transform =transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    img_tensor = transform(img)
    img_numpy = img_tensor.numpy()
    return img_numpy