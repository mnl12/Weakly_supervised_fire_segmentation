import numpy as np
import tensorflow as tf
from PIL import Image
from models import segmentation_network
from my_metrics import binary_miou_metric, binary_numpy_iou, my_binary_numpy_iou
from generators import fire_image_generator, smoke_image_generator
from utils import create_tf_binary_mask, binary_labels_onehot, binary_labels_onehot_tf, tf_compute_weights
import os


class label_supervision(tf.keras.Model):
    def __init__(self, lmodel):
        super(label_supervision, self).__init__()
        self.model=lmodel

    def call(self, input):
        return self.model(input)


shuffle=1
fire_img=0
#fire optimal cam_thre=.45 cam_thre_mid=.55
cam_thre=.05
cam_thre_mid=.05
BATCH_SIZE = 12
IMAGE_SIZE=(256,256)
ANY_SIZE=1
model_name='fcn-16'
base_model_name='vgg'
MASK_SIZE=IMAGE_SIZE
ADD_FEAT=3
n_classes=1
EPOCHS = 100
normalize=1
if fire_img==1:
    train_img_dir='../dataset/test_weakly_2/rgb/rgb/'
    train_mask_dir='../dataset/test_weakly_2/masked/masked/'
    test_ids = next(os.walk(train_img_dir))[2]
    test_generator=fire_image_generator(train_img_dir, train_mask_dir, test_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, shuffle)
else:
    test_img_dir = '../dataset/Smoke/test/images/'
    test_mask_dir = '../dataset/Smoke/test/masks/'
    train_smoke_dir = '../dataset/Smoke/train/Smoke/'
    train_nosmoke_dir = '../dataset/Smoke/train/No-smoke/'
    test_generator = smoke_image_generator(test_img_dir, test_mask_dir, train_smoke_dir, train_nosmoke_dir, BATCH_SIZE,
                                           IMAGE_SIZE, MASK_SIZE, 1, shuffle, 0)

OUTPUT_CHANNELS=n_classes
model1 = segmentation_network(base_model_name,model_name, OUTPUT_CHANNELS, IMAGE_SIZE,ADD_FEAT)
model=label_supervision(model1)
if fire_img==1:
    checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
else:
    checkpint_folder = model_name+'-'+base_model_name+'-smoke-'+str(IMAGE_SIZE[0])


checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-0020.ckpt'
model.load_weights(checkpoint_path)

num_epochs=test_generator.__len__()
#print('Number of samples is:{}'. format(test_generator.num_of_samples()))
#num_epochs=6

label_acc_metric = tf.keras.metrics.BinaryAccuracy()
recall_m=tf.keras.metrics.Recall()
iou_v=np.zeros((num_epochs))
iou_v2=np.zeros((num_epochs))

for i in range(num_epochs):
    sample_image_batch, [sample_label_batch, sample_mask_batch] = test_generator.__getitem__(i)
    pred_label, pred_mask, pred_sig = model(sample_image_batch, training=False)
    sample_label_onehot=binary_labels_onehot_tf(pred_label)

    pred_mask=create_tf_binary_mask(pred_mask, cam_thre)
    pred_sig = create_tf_binary_mask(pred_sig, cam_thre_mid)

    pred_mask=np.multiply(sample_label_onehot.numpy()[...,np.newaxis,np.newaxis], pred_mask)
    pred_sig=np.multiply(sample_label_onehot.numpy()[...,np.newaxis,np.newaxis], pred_sig)

    label_acc_metric.update_state(sample_label_batch, pred_label)
    label_acc_v=label_acc_metric.result().numpy()
    recall_m.update_state(sample_mask_batch, pred_mask)
    iou_v[i]=my_binary_numpy_iou(sample_mask_batch, pred_mask, .5)
    iou_v2[i]=my_binary_numpy_iou(sample_mask_batch, pred_sig, .5)

iou=np.mean(iou_v[:-1])
iou_tf=np.mean(iou_v2[:-1])
print('ACC is {}, Recall is {}, and the IOU is {}, and IOU mid level is {}'.format(label_acc_v, recall_m.result().numpy(), iou, iou_tf))
