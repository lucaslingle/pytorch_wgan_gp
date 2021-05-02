# pytorch_wgan_gp

This is a Pytorch implementation of the WGAN-GP ResNet model for CIFAR-10, from Gulrajani et al., 2017 - 'Improved Training of Wasserstein GANs'.

The implementation seems to be working, but it will require about 70 days to run on my local machine. If I train it on a GPU, I will update the readme 
with an approximate duration and cost for doing so.

Update: Training 1000 generator steps on an NVIDIA V100 GPU takes about 14 minutes by my count, so 100,000 steps will require 100 * 14 minutes, or approximately 23 hours.
The total cost, assuming an instance price of $1.81/hr, is approximately $42. 

