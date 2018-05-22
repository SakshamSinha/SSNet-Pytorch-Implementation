# Pytorch installation instructions

This is a step-by-step instruction for installing and running the code in PyTorch for the final project in the course CS7GV1 - Computer Vision

## Data

You can download the data from the following links
* [Training images](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
* Testing images (to be updated soon)

## Prerequisites

```
Ubuntu (14 or higher)
Cuda 8
cuDNN (optionally used to accelerate training)
python-pip
virtualenv (optional)
```
## Installation

### Step 1 : Installing Cuda
Download CUDA 8 and cuDNN library from the following links. The default website takes one to CUDA 9. The previous versions can be found in "Legacy Releases" under "Additional Resources".
* [Cuda 8](https://developer.nvidia.com/cuda-80-ga2-download-archive)
* [cuDNN 7 for cuda 8](https://drive.google.com/a/tcd.ie/file/d/1JNKUnIRbAnZ49wSiJgou4zz9CQiamR8J/view?usp=sharing)

For CUDA, follow the installation instructions from the website. For cuDNN, they ask you to create an account in order to download the library. You don't have to since we have downloaded it for you. You can just follow the installation intructions mentioned [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
### Step 2 : Creating a virtual environment(optional)
[Why use virtual environment ?](https://pythontips.com/2013/07/30/what-is-virtualenv/)

Run the following commands
```
virtualenv -p /path/to/python2.7 project_name
source project_name/bin/activate
```
"/path/to/python2.7" is usually /usr/bin/python2.7. 
Once you activate a virtual environment, you will be able to use only the local packages. Check this link if you wish to know how to use the global packages. 

[Make virtualenv inherit specific packages from your global site-packages
](https://stackoverflow.com/questions/12079607/make-virtualenv-inherit-specific-packages-from-your-global-site-packages)
### Step 3 : Installing pytorch
```
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
pip install torchvision
```
## Running the script
First, set up the data using the script process_data.sh. It creates a folder called 'data', which contains 'train' and 'val' folders. Each of those folders contain 200 sub-folders containing the training and validation data.  You have to do this only once though.
```
./process_data.sh /path/to/the/folder/where/you/downloaded/TinyImageNet/
```
In Models.py, there are two networks available. The Demo network which accepts the input in its original size i.e 64\*64. You run the demo with the following command.

```
python Example.py --cuda --datapath /path/to/data/ --model Demo --aug H_FLIP
```
the '--aug' option decides the augmentation strategy, which is a random horizontal flip in this case. 

For running the pre-trained ResNet18, use this command

```
python Example.py --cuda --pretrained  --datapath /path/to/data/ --model ResNet18 --aug SCALE_H_FLIP
```

Please note the augmentation strategy used is different. ResNet18 has a very deep architecture and the feature map size becomes zero at a certain layer for an input size of 64\*64. Hence we resize the input to 224\*224. 

Please check the code for additional parameters.

## Testing the model

Check out the commented section of Example.py, at the very end. The data needs to be in the following order for pytorch specifications
```
root_folder/val/test_images/*.jpg

```
The code works for any number of images in the folder. Check the kaggle competition page for downloading the test data.
