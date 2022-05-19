# vis_assignment3
Link to github repo: https://github.com/AndersRoen/vis_assignment3.git
## Assignment 3 description

In the previous assignment, you built simple set of classifiers to work with the CIFAR10 data set. Those classifiers were fairly simple, insofar as we just flattened the images into a 1D array of features and fed that into either a LogisticRegression Classifier or a fully-connected, feed-forward neural network.

This approach - converting to grey scale, flattening images - means that we lose a lot of potentially meaningful information in the data. We've also seen, though, that Convolutional Neural Networks are specifically designed for working with image data, allowing us to retain more of the original image. Particularly deep, complex CNNs that have been pretrained on a huge amount of data can be used as a feature extraction tool, converting our images into dense vector representations. This process is known as transfer learning.

In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction.

Your .py script should minimally do the following:

    Load the CIFAR10 dataset
    Use VGG16 to perform feature extraction
    Train a classifier
    Save plots of the loss and accuracy
    Save the classification report
## 
    Use argparse() to allow users to define specific hyperparameters in the script.
        This might include e.g. learning rate, batch size, etc
    The user should be able to define the names of the output plot and output classification report from the command line

## Methods
To get this script to run optimally, you should run the ```setup.sh``` script.
This assignment uses transfer learning, by building a classifier with transfer learning principles, namely utilizing a pretrained model - in this case VGG16 to extract features.
The script, ```assignment_3.py``` works like this:
First, the script defines the pretrained model. The model uses a sgd optimizing algorithm with a learning schedule with user-defined hyperparameters. 
Then, the model loads in the ```cifar10``` dataset, splitting it into ```train``` and ```test``` data.
The model is then trained, with a user-defined ```batch_size``` and ```epochs```. The loss and accuracy across epochs are then plotted and saved to the ```out``` folder, with a user-defined filename.
Lastly, the script generates a classification report and saves it to the ```out``` folder with a user-defined filename.

## Usage
To run this script, point the command line to the ```vis_assignment3``` folder.
The script has 7 command line arguments that are needed to run the script:
```--eps``` defines how many epochs you want the model to train for. I trained the model for 10 epochs.
```--vn``` defines the filename of the plot file, such as ```example_plot.png```
```--learn_rate``` defines the learning rate of the model. I used a learning rate of 0.01.
```--dec_steps``` defines the decay steps of the model. I set it to 1000
```--dec_rate``` defines the decay rate of the model. I set it to 0.9
```b_size``` defines the batch size of the model. I set it to 128.
```rep_name``` defines the filename of the classification report.

## Results
This model yielded an accuracy of 53%. Fiddling around with the hyperparameters could potentially improve the model, but I wasn't able to improve it by that much. However, when you compare the accuracy of the transfer learning model to the neural network from assignment 2, which yielded around 35% accuracy, there is a quite significant rise in accuracy. This model thus shows how transfer learning can take machine learning further from "normal" neural networks.


