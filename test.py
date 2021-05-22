from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from PIL import Image
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors
from models import segmentation_network
from my_metrics import binary_numpy_iou, my_binary_numpy_iou, other_accuracy, binary_miou_metric, general_iou
from utils import create_mask_tr, create_tf_binary_mask
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
fire_img=0
#fire optimal  cam_thre=0.75 cam_thre_mid=.45
cam_thre=0.45
cam_thre_mid=.25
BATCH_SIZE = 32
IMAGE_SIZE=(256,256)
n_classes=1
ADD_FEAT=3
model_name='fcn-16'
base_model_name='vgg'
if fire_img==1:
    db_path='../dataset/'
    img_folder='test_weakly/rgb/rgb/'
    mask_folder='test_weakly/masked/masked/'
    image_name='022'
    img_path=db_path+img_folder+image_name+'.png'
    mask_path=db_path+mask_folder+image_name+'.png'
else:
    test_img_dir = '../dataset/Smoke/test/images/'
    test_mask_dir = '../dataset/Smoke/test/masks/'
    image_name='L_060.png'
    img_path = test_img_dir + image_name
    mask_path=test_mask_dir + image_name

sample_image = np.array(Image.open(img_path).convert('RGB').resize(IMAGE_SIZE), dtype=float)/255.0
sample_image_input=sample_image
sample_mask = np.array(Image.open(mask_path).convert('L').resize(IMAGE_SIZE, Image.NEAREST), dtype=float)/255.0

#Model define
OUTPUT_CHANNELS=n_classes
model1 = segmentation_network(base_model_name, model_name, OUTPUT_CHANNELS, IMAGE_SIZE,ADD_FEAT)
model=label_supervision(model1)

if fire_img==1:
    checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
else:
    checkpint_folder = model_name+'-'+base_model_name+'-smoke-'+str(IMAGE_SIZE[0])

checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-0040.ckpt'
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


[predlabel0, pred_mask, pred_mask_sig]=model.predict(sample_image_input[np.newaxis, ...])
pred_binary_mask=create_tf_binary_mask(pred_mask, cam_thre)
pred_mask_vis=create_tf_binary_mask(pred_mask_sig, cam_thre_mid)
pred_binary_mask_numpy=pred_binary_mask.numpy()
miou_val=binary_numpy_iou(sample_mask[np.newaxis,...,np.newaxis], pred_binary_mask.numpy())
miou_2=my_binary_numpy_iou(sample_mask[np.newaxis, ...,np.newaxis], pred_mask_vis.numpy(), cam_thre)
acc=tf.keras.metrics.Accuracy()
recall_m=tf.keras.metrics.Recall()
acc.update_state(sample_mask[np.newaxis,...,np.newaxis], pred_binary_mask)
recall_m.update_state(sample_mask[np.newaxis,...,np.newaxis], pred_binary_mask)
print('ACC : {}, Recall: {}, IOU final: {}, IOU middle {}'.format(acc.result().numpy(), recall_m, miou_val, miou_2))
display([sample_image,  np.squeeze(pred_binary_mask), np.squeeze(pred_mask_vis)])

#Saving images
Image.fromarray((np.squeeze(pred_binary_mask)*255).astype(np.uint8)).save('Results/mid_vis/cam_'+image_name+'.png')
Image.fromarray((np.squeeze(pred_mask_vis)*255).astype(np.uint8)).save('Results/mid_vis/vis_'+image_name+'.png')




