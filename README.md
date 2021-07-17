Weakly supervised fire segmentation with intermediate layer visualization
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 

<h1>Method description</h1>
It is a fire segmentation method that uses only image labels (fire or not-fire) in the training. 
In the trainining, the method first train a classificaton network for fire and not-fire. Then initial masks are obtained by visulaizing the CNN intermidate layers. These masks are used as groundtruth mask to train a second network which is a segmentation network.

<h1>The code</h1>
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
  <tr>
    <td>Original image</td>
    <td>groundtruth mask</td>
    <td>CAM</td>
    <td>mid-layer visulaization</td>
    <td>Proposed</td>
  </tr>
 </table>
  



