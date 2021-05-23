# Weakly_supervised_fire_segmentation
Weakly supervised fire segmentation with intermediate layer visualization


The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 
It contains two networks to be trained: train2.py is used to train the first network to obtain initial visulization masks
train.py trains a deeplabv3 pixel-supervised segmentation CNN with the intial masks
