# Weakly_supervised_fire_segmentation
Weakly supervised fire segmentation with intermediate layer visualization
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 
<br />
It contains two networks to be trained: 
<ul>
  <li>train2.py is used to train the first network to obtain initial visulization masks</li>
<li>train.py trains a deeplabv3 pixel-supervised segmentation CNN with the intial masks</li>
  </ul>
  
