# MFV_on_PyTorch

## About the project

"Updating the Multifaceted Feature Visualization algorithm from Caffe to PyTorch for evaluating interpretability of contemporary neural networks"

In this project, our contribution involves several steps: first, we run the authors' code and identify the exact versions of libraries and Caffe framework used for MFV algorithm. Next, we implement the MFV algorithm using PyTorch framework and libraries as needed. Finally, we apply the updated MFV algorithm to CNNs, including AlexNet, VGG16, GoogLeNet and SqueezeNet, in order to demonstrate its applicability and relevance in the current landscape of deep learning research.

## Project structure

The folder `alexnet_caffe2pytorch` contains a Dockerfile for converting the AlexNet model version used by the authors from Caffe to PyTorch and a Python file - a conversion script.

The `environment` folder contains an environment with all the dependencies required to run the MFV code on Caffe locally.

The `MFV_for_neural_networks.ipynb` file is used to run the MFV algorithm on PyTorch for modern neural networks.

The `alculating_mean_images.ipynb` file contains the code necessary for generating the "mean images" for the MFV algorithm.

`net.pt` refers to the weights of the AlexNet model version used by the authors, translated to PyTorch.
