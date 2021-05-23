Weakly supervised fire segmentation with intermediate layer visualization
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 


It contains two networks to be trained: 
<ul>
  <li>train2.py is used to train the first network to obtain initial visulization masks</li>
<li>train.py trains a deeplabv3 pixel-supervised segmentation CNN with the intial masks</li>
  </ul>
  
To Evalue the performance on the test set run: evaluate.py for evaluating initial masks, and evaluate_segment.py for evaluating the second network and the final performance


In order to get output of image examples run: test.py for intial masks and test_segment.py for the final network
  
