Weakly supervised fire segmentation with intermediate layer visualization
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 


It contains two networks to be trained: 
<br />
train2.py is used to train the first network to obtain initial visulization masks <br />
train.py trains a deeplabv3 pixel-supervised segmentation CNN with the intial masks

  
To Evalue the performance on the test set run: evaluate.py for evaluating initial masks, and evaluate_segment.py for evaluating the second network and the final performance


In order to get output of image examples run: test.py for intial masks and test_segment.py for the final network

<table>
  <td><img src='https://github.com/mnl12/Weakly_supervised_fire_segmentation/blob/main/images/019.png' width=150></td>
  <td><img src='https://github.com/mnl12/Weakly_supervised_fire_segmentation/blob/main/images/019_mask.png' width=150></td>
  <td><img src='https://github.com/mnl12/Weakly_supervised_fire_segmentation/blob/main/images/cam_019.png' width=150></td>
  <td><img src='https://github.com/mnl12/Weakly_supervised_fire_segmentation/blob/main/images/vis_019.png' width=150></td>
  <td><img src='https://github.com/mnl12/Weakly_supervised_fire_segmentation/blob/main/images/segment_019_1.png' width=150></td></tr>
  <tr><td></td>
    <td>Original image</td>
    <td>groundtruth mask</td>
    <td>CAM</td>
    <td>visulaization of intermediate layers</td>
    <td>Proposed</td>
  </tr>
 </table>
  



