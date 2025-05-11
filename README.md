<h1>Weakly supervised fire segmentation with intermediate layer visualization</h1>
<br />
The code corrosponds to the paper "Weakly-supervised fire segmentation by visualizing intermediate CNN layers" 

<h2>Method description</h2>
It is a fire segmentation method that uses only image labels (fire or not-fire) in the training. 
The method first trains a classificaton network for fire or not-fire. Then initial masks are obtained by visulaizing the CNN intermidate layers. These masks are used as groundtruth masks to train a second network which is a segmentation network.

<h2>Code</h2>
<h4>Installation</h4>
 <br />
<pre><code>pip install -r requirements.txt</code></pre>


<h4>Dataset</h4>
Dataset can be downloaded from:
https://drive.google.com/file/d/1tWWFhEaoGaitLBOBjT-78ZJEcGW_kaag/view?usp=sharing

<h4>File descriptions:</h4>
train.py: trains a deeplabv3 segmentation CNN with the intial masks as groundtruth <br />
evaluate.py: evalues the initial masks on the test set <br />
evaluate_segment.py: evaluates the final results on the test set
test.py: gives the initial segmented mask given an input image  <br />
test_segment.py: gives the final segmented mask for a given input image

<p> </p>
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
    <td>Final output</td>
  </tr>
 </table>
  



