from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from PIL import Image
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors
from models import segmentation_network
from my_metrics import binary_numpy_iou, my_binary_numpy_iou, other_accuracy, binary_miou_metric, general_iou
from utils import create_mask_tr, create_tf_binary_mask, binary_labels_onehot_tf
from IPython.display import clear_output
import os


class label_supervision(tf.keras.Model):
    def __init__(self, lmodel):
        super(label_supervision, self).__init__()
        self.model=lmodel

    def call(self, input):
        return self.model(input)




def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if len(display_list[i].shape)==2:
            img_mode='L'
            plt.imshow(Image.fromarray((display_list[i]).astype(np.uint8), mode=img_mode), cmap='gray')
        else:
            img_mode='RGB'
            plt.imshow(Image.fromarray((display_list[i] * 255).astype(np.uint8), mode=img_mode))

        plt.axis('off')
    plt.show()


seed=1
cam_thre=0.75
cam_thre_mid=.45
segment_step=1
mid_lev_train=1
BATCH_SIZE = 32
IMAGE_SIZE=(256,256)
n_classes=1
ADD_FEAT=3
model_name='fcn-16'
base_model_name='vgg'
db_path='../dataset/'
img_folder='test_weakly/rgb/rgb/'
mask_folder='test_weakly/masked/masked/'
image_name='019'
img_path=db_path+img_folder+image_name+'.png'
mask_path=db_path+mask_folder+image_name+'.png'



sample_image = np.array(Image.open(img_path).convert('RGB').resize(IMAGE_SIZE), dtype=float)/255.0
sample_image_input=sample_image
sample_mask = np.array(Image.open(mask_path).convert('L').resize(IMAGE_SIZE, Image.NEAREST), dtype=float)/255.0

#Model define

model = segmentation_network('resnet', 'deeplab', 1, IMAGE_SIZE, ADD_FEAT)
checkpoint_folder = model_name + '-' + base_model_name + '-' + str(mid_lev_train) + str(IMAGE_SIZE[0])
checkpoint_path = 'checkpoints/' + checkpoint_folder + '/cp-any448-{j:04d}.ckpt'
model.load_weights(checkpoint_path)



def create_binary_mask(pred_input_mask, thre):
    pred_mask=np.ones_like(pred_input_mask)
    pred_mask[pred_input_mask<thre]=0
    return pred_mask[0]

def create_mask(pred_mask):
    pred_mask1=tf.argmax(pred_mask, axis=-1)
    pred_mask=pred_mask1[0]
    pred_mask=labels2colors(pred_mask, c_map)
    return pred_mask

tst_acc=tf.keras.metrics.BinaryAccuracy()
[predlabel0, pred_mask, pred_mask_sig]=model.predict(sample_image_input[np.newaxis, ...])
pred_sig = binary_labels_onehot_tf(pred_mask_sig)
display([np.squeeze(sample_mask), np.squeeze(pred_sig)])
tst_acc.update_state(sample_mask, pred_sig)
iou_v_test = my_binary_numpy_iou(sample_mask[np.newaxis, ...], pred_sig.numpy()[np.newaxis,...], .5)


#Saving images
Image.fromarray((np.squeeze(pred_sig)*255).astype(np.uint8)).save('Results/segmented_results/cam_'+image_name+'_'+str(mid_lev_train)+'.png')
#Image.fromarray((np.squeeze(pred_mask_vis)*255).astype(np.uint8)).save('Results/seg_mid_vis/vis_'+image_name+'.png')




