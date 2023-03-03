# Importing required Python modules
import utils
import model_helper as mh
import torch
import os
from torch import nn, optim

# Defining the main function below
def main():
    
    # Gets the collection of arguments in args
    args = utils.get_train_arguments()
    
    # Loads the datasets and creates dataloaders
    image_datasets, dataloaders = utils.load_datasets(args.data_dir)
    
    # Creating the model
    model = mh.create_model(args.arch, args.hidden_units, len(image_datasets['train'].class_to_idx))
    
    # Defining criterion
    criterion = nn.NLLLoss()

    # Defining optimizer only for the classifier part
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)
    
    # Defining what device will be used for training
    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        print('Training using GPU...')
        device = 'cuda'
    elif args.gpu and not torch.cuda.is_available():
        print('GPU is not available....Training using CPU...')
    else:
        print('Training using CPU...')
        
    # Training the model
    mh.train_model(model, dataloaders, args.epochs, device, criterion, optimizer)
    
    print('Testing on test set....')
    
    # Testing the model on test set
    mh.test_model(model, dataloaders, criterion, device)
    
    # Definining the checkpoint file name
    path = 'checkpoint_' + args.arch + '_epochs_' + str(args.epochs) + '.pth'
    
    # Checking if save directory is present or not
    if os.path.exists(args.save_dir):
        path = args.save_dir + '/' + path
    elif not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        path = args.save_dir + '/' + path
    
    # Saving the checkpoint
    model.to('cpu')
    checkpoint = {'model': args.arch,
                 'epochs': args.epochs,
                 'input_size': model.classifier[0].in_features,
                 'output_size': model.classifier[-2].out_features,
                 'hidden_layer': args.hidden_units,
                 'class_to_idx': image_datasets['train'].class_to_idx,
                 'state_dict': model.state_dict(),
                 'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, path)
    
# Calls main function to run the program
if __name__ == '__main__':
    main()