# Image_Classifier_Udacity
This is the final project in the **Udacity Nanodegree** on [AI Programming with Python](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).
The project has two parts:
1. Training a pretrained ImageNet model like VGG using transfer learning on [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) containing images of 102 different species of flowers to classify new images of flowers.
2. Creating command line applications that can be used to train the classifier model with several execution options and use the trained model for inference.

## Part 1: Flower Image Classifier
----------------------------------
This part includes a Jupyter Notebook `Image Classifier Project.ipynb` where a `densenet121` model is trained using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) containing images of 102 species of flowers.
The mapping of category(1-102) to names of the flowers is present in `cat_to_name.json` file.

### Usage
---------
1. Clone this repository and open the file named `Image Classifier Project.ipynb`.
2. Run the cell containing the following code to import necessary libraries: 
```python
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt 
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sb
```
3. Run the cell within the section called **Label mapping**.
4. Run the cell in the section **Loading the checkpoint**. This loads a the saved model present in the file `checkpoint/checkpoint_densenet121.pth`.
5. Run all the cells in the **Inference for classification** section and the **Class Prediction** section.
6. Now to predict the species of the flower, provide the path of the flower image in the last cell in **Sanity Checking** section. Replace the path in the `image_path` variable with the path of your image.
```python
# Display an image along with the top 5 classes
fig, (ax1,ax2) = plt.subplots(figsize=(5, 10), nrows= 2)
image_path = 'Provide the path to the image'
img = process_image(image_path)
title = cat_to_name[image_path.split('/')[2]]
imshow(img, ax = ax1, title= title);

probs, classes = predict(image_path, model)

ticks = [cat_to_name[key] for key in classes]
sb.barplot(x=probs, y = ticks, ax = ax2, color=sb.color_palette()[0])
plt.show()
```

### Accuracy
------------
Accuracy on validation set: 0.901 (90.1 %)\
Accuracy on test set: 0.886 (88.6 %)

### Example Output
------------------
The output will provide the original image along with **top 5** predicted classes. For example:

<img src='assets/inference_example.png' width=300px>

## Part 2: Command Line Applications
------------------------------------
This part includes two Python command line applications `train.py` and `predict.py`.
* `train.py` is used to train an ImageNet classifier model among `vgg13`, `densenet121`, `resnet18` on a given Image dataset and save the trained model.
* `predict.py` is used to load the trained image classifier model obtained from `train.py` and classify a given image.

### Install Requirements
------------------------
Clone this repository and run the following commands in command line:
```
cd Image_Classifier_Udacity
pip install -r requirements.txt
```

### Usage for `train.py`
------------------------
```
Usage:
    python train.py [options] [<path>]

Arguments:
    <path>                                  The path to the directory containing datasets (Required)

Options:
    -h, --help                              Shows help message for this command
    --save_dir <path>                       The path to the directory for saving the model (Default: current directory)
    --arch {vgg13,densenet121,resnet18}     The name of the pretrained model to be used (Default: densenet121)
    --learning_rate <float>                 The learning rate of the model (Default: 0.003)
    --hidden_units <list>                   The list of hidden units for the network (Default: [512, 256])
    --epochs <int>                          The number of epochs to be trained for (Default: 10)
    --gpu                                   Use if you want to train with GPU
```

### Example
-----------
```
$ python train.py --save_dir checkpoint --arch vgg13 --learning_rate 0.01 --hidden_units [512,256,128] --epochs 20 --gpu flowers
```
### Usage for `predict.py`
--------------------------
```
Usage:
    python predict.py [options] [<path 1>] [<path 2>]

Arguments:
    <path 1>                                The path to the image file that you want to classify (Required)
    <path 2>                                The path to the saved model checkpoint obtained from train.py (Required)

Options:
    -h, --help                              Shows help message for this command
    --top_k <int>                           The number of top predicted classes (Default: 1)
    --category_names <path>                 Path to the JSON file containing mapping of categories to real name (Default: None)
    --gpu                                   Use if you want to predict with GPU
```
### Example
-----------
```
$ python predict.py --top_k 5 --category_names cat_to_name.json flowers/test/70/image_05324.jpg checkpoint/checkpoint_densenet121.pth
Predicting using CPU...
1 Flower Name: tree poppy, Probability: 0.996
2 Flower Name: pincushion flower, Probability: 0.002
3 Flower Name: colt's foot, Probability: 0.001
4 Flower Name: prince of wales feathers, Probability: 0.000
5 Flower Name: carnation, Probability: 0.000
```

## License
----------
The License for this project can be found in the [LICENSE](LICENSE) file.