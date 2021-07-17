<h1>Weakly supervised fire segmentation with intermediate layer visualization</h1>
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 

<h2>Method description</h2>
It is a fire segmentation method that uses only image labels (fire or not-fire) in the training. 
In the trainining, the method first train a classificaton network for fire and not-fire. Then initial masks are obtained by visulaizing the CNN intermidate layers. These masks are used as groundtruth mask to train a second network which is a segmentation network.

<h2>The code</h2>
It contains two networks to be trained: 
<br />
train2.py: trains the first network to obtain initial visulization masks <br />
train.py: trains a deeplabv3 segmentation CNN with the intial masks as groundtruth

  
evaluate.py: evalues the initial masks on the test set <br />
evaluate_segment.py: evaluates the final results on the test set

 test.py: gives the initial segmented mask given an input image  <br />
 test_segment.py: gives the final segmented mask for a given input image

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
  



