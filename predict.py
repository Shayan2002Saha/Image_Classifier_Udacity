# Importing required Python modules
import utils
import model_helper as mh
import torch
import os
import json
from torch import nn, optim

# Defining the main function below
def main():
    
    # Gets the collection of arguments in args
    args = utils.get_predict_arguments()
    
    if os.path.exists(args.input) and os.path.exists(args.checkpoint):
        
        # Loading the checkpoint and rebuilding the model
        model = utils.load_checkpoint(args.checkpoint)
        
        # Processing the image
        img = torch.from_numpy(utils.process_image(args.input))
        img.unsqueeze_(0)
        
        # Defining what device will be used for predicting
        device = 'cpu'
        if args.gpu and torch.cuda.is_available():
            print('Predicting using GPU...')
            device = 'cuda'
        elif args.gpu and not torch.cuda.is_available():
            print('GPU is not available....Predicting using CPU...')
        else:
            print('Predicting using CPU...')
        
        # Moving model and image to the selected device
        model.to(device)
        img = img.to(device)
        
        # Predicting
        model.eval()
        with torch.no_grad():
            ps = torch.exp(model.forward(img))
        top_p, top_label = ps.topk(args.top_k, dim=1)
        top_p, top_label = top_p.to('cpu'), top_label.to('cpu')
        idx_to_class = {value : key for key, value in model.class_to_idx.items()}
        top_class = [idx_to_class[label] for label in top_label.numpy().reshape(args.top_k)]
        top_p = top_p.numpy().reshape(args.top_k)
       
        # Getting the real class
        if args.category_names != None and os.path.exists(args.category_names):
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            top_class = [cat_to_name[key] for key in top_class]
        elif args.category_names != None and not os.path.exists(args.category_names):
            print("category_names file does not exist....Using folder labels as class")
        
        # Printing the result
        for rank, (prob, cat) in enumerate(zip(top_p, top_class)):
            print(f"{rank + 1} Flower Name: {cat}, Probability: {prob:.3f}")
            
    elif not os.path.exists(args.input):
        print("Image file does not exist")
        
    else:
        print("Checkpoint file does not exist")
    
# Calls main function to run the program
if __name__ == '__main__':
    main()