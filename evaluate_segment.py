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
from generators import fire_image_generator, smoke_image_generator
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
fire_img=1
cam_thre=0.75
cam_thre_mid=.45
segment_step=1
mid_lev_train=1
BATCH_SIZE = 32
IMAGE_SIZE=(256,256)
n_classes=1
ADD_FEAT=3
MASK_SIZE=IMAGE_SIZE
shuffle=1
model_name='fcn-16'
base_model_name='vgg'
db_path='../dataset/'
img_folder='test_weakly/rgb/rgb/'
mask_folder='test_weakly/masked/masked/'
image_name='013'
img_path=db_path+img_folder+image_name+'.png'
mask_path=db_path+mask_folder+image_name+'.png'

#Get fire images
if fire_img==1:
    train_img_dir='../dataset/test_weakly_2/rgb/rgb/'
    train_mask_dir='../dataset/test_weakly_2/masked/masked/'
    normalize=1
    test_ids = next(os.walk(train_img_dir))[2]
    test_generator=fire_image_generator(train_img_dir, train_mask_dir, test_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, shuffle)

else:
    train_smoke_dir = '../dataset/Smoke/train/Smoke/'
    train_nosmoke_dir = '../dataset/Smoke/train/No-smoke/'
    test_img_dir = '../dataset/Smoke/test/images/'
    test_mask_dir = '../dataset/Smoke/test/masks/'
    test_generator = smoke_image_generator(test_img_dir, test_mask_dir, train_smoke_dir, train_nosmoke_dir, BATCH_SIZE,
                                           IMAGE_SIZE, MASK_SIZE, 1, shuffle, 0)

num_epochs=test_generator.__len__()
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

acc_metric = tf.keras.metrics.BinaryAccuracy()
iou_v=np.zeros((num_epochs))

for i in range(num_epochs):
    sample_image_batch, [sample_label_batch, sample_mask_batch] = test_generator.__getitem__(i)
    [predlabel0, pred_mask, pred_mask_sig]= model(sample_image_batch, training=False)
    pred_sig = binary_labels_onehot_tf(pred_mask_sig)
    iou_v[i] = my_binary_numpy_iou(sample_mask_batch, pred_sig, .5)
    acc_metric.update_state(sample_mask_batch, pred_sig)


iou=np.mean(iou_v[:-1])
print('ACC is {}, and the IOU is {}'.format(acc_metric.result().numpy(), iou))




