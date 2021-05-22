from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from IPython.display import clear_output
import random
import os
from PIL import Image
from my_metrics import binary_miou_metric, others_iou, me_others_iou, general_iou, general_iou_ignore, binary_numpy_iou
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors, onehot2mask
from generators import fire_image_generator, smoke_image_generator
from models import segmentation_network
from utils import display, create_mask2, create_mask_bg, create_mask_tr, create_binary_mask, create_tf_binary_mask



class label_supervision(tf.keras.Model):
    def __init__(self, lmodel, opt, reg_consis):
        super(label_supervision, self).__init__()
        self.model=lmodel
        self.opt=opt
        self.cat_ce_loss = tf.keras.losses.CategoricalCrossentropy()
        self.b_ce_loss=tf.keras.losses.BinaryCrossentropy()
        self.label_metric=tf.keras.metrics.Accuracy()
        self.miou_metric=binary_miou_metric
        self.test_iter = 0
        self.reg=reg_consis

    def det_reg(self, iter, init_val, end_val, num_iter):
        return init_val+(end_val-init_val)*iter/num_iter

    def call(self, input):
        return self.model(input)

    def train_step(self, batch_data):
        print(self.test_iter)
        self.test_iter+=1
        #self.reg = self.det_reg(self.test_iter, .00005, 1, 30)
        k_rot = random.randrange(1, 4)
        sample_image_batch, [labels_batch, sample_mask_batch] = batch_data
        sample_image_batch_rot = tf.image.rot90(sample_image_batch, k_rot)
        acd, dfv, pred_mask_sig = self.model(sample_image_batch, training=False)
        pred_mask_sup=create_tf_binary_mask(pred_mask_sig, .6)

        with tf.GradientTape() as tape:
            pred_label, pred_mask, pred_mask_sig = self.model(sample_image_batch, training=True)

            #consistency
            #pred_label_rot, pred_mask_rot, pred_jnk = self.model(sample_image_batch_rot, training=True)
            #pred_mask_360 = tf.image.rot90(pred_mask_rot, 4 - k_rot)
            #consistency_loss = tf.keras.losses.mse(pred_mask, pred_mask_360)

            # pred_mask_bg = create_mask_bg(pred_mask)

            label_loss = self.b_ce_loss(labels_batch, pred_label)
            segment_loss = self.b_ce_loss(sample_mask_batch, pred_mask)
            #sup_loss=tf.keras.losses.mse(pred_mask, pred_mask_sig)

           # overall_loss = label_loss+self.reg*sup_loss+0.01*consistency_loss
            overall_loss = label_loss


        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(overall_loss, trainable_vars)
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        v_label_metric = self.label_metric.update_state(labels_batch, pred_label)

        #Uncomment for fire
        #pred_mask=create_tf_binary_mask(pred_mask, .4)
        #v_segment_iou = self.miou_metric(sample_mask_batch, pred_mask)

        return {"Label_loss": label_loss, "Segmentation_loss": segment_loss, "label_acc": self.label_metric.result()}

    @property
    def metrics(self):
        return [self.label_metric]

    def test_step(self, data):
        sample_image_batch, [labels_batch, sample_mask_batch] = data
        pred_label, pred_mask, pred_sig = self.model(sample_image_batch, training=False)

        label_loss = self.b_ce_loss(labels_batch, pred_label)
        segment_loss = self.b_ce_loss(sample_mask_batch, pred_mask)

        pred_mask = create_tf_binary_mask(pred_mask, .6)
        v_segment_iou = self.miou_metric(sample_mask_batch, pred_mask)
        return {"Label_loss":label_loss, "Segmentation_loss":segment_loss, "Segment_IOU":v_segment_iou}



seed=1
fire_db=0
extra_train=0
RAND_TRAIN_VAL=0
shuffle=1
augment=0
DISPLAY_THRE=0
BATCH_SIZE = 12
IMAGE_SIZE=(256,256)
model_name='fcn-16'
base_model_name='vgg'
MASK_SIZE=IMAGE_SIZE
n_classes=1
EPOCHS = 50
ADD_FEAT=3
reg_consis=0
cam_thre=.6
reg_ind= 1 if reg_consis>0 else 0

if fire_db==1:
    db_path='../dataset/'
    img_path_train='Weakly_fire/images/'
    mask_path_train='Weakly_fire/masks/'
    img_path_test='test/rgb/rgb/'
    mask_path_test='test/masked/masked/'
    train_img_dir=db_path+img_path_train
    train_mask_dir=db_path+mask_path_train
    test_img_dir=db_path+img_path_test
    test_mask_dir=db_path+mask_path_test

    train_ids = next(os.walk(train_img_dir))[2]
    test_ids = next(os.walk(test_img_dir))[2]
    train_generator = fire_image_generator(train_img_dir, train_mask_dir, train_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE,
                                           1, shuffle)
    test_generator = fire_image_generator(test_img_dir, test_mask_dir, test_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1,
                                          shuffle)

else:
    test_img_dir = '../dataset/Smoke/test/images/'
    test_mask_dir = '../dataset/Smoke/test/masks/'
    train_smoke_dir = '../dataset/Smoke/train/Smoke/'
    train_nosmoke_dir = '../dataset/Smoke/train/No-smoke/'
    train_generator = smoke_image_generator(test_img_dir, test_mask_dir, train_smoke_dir, train_nosmoke_dir, BATCH_SIZE,
                                            IMAGE_SIZE, MASK_SIZE, 1, shuffle, 1)
    test_generator = smoke_image_generator(test_img_dir, test_mask_dir, train_smoke_dir, train_nosmoke_dir, BATCH_SIZE,
                                           IMAGE_SIZE, MASK_SIZE, 1, shuffle, 0)




BUFFER_SIZE = 1000
sample_image_batch, [sample_label_batch, sample_mask_batch]=train_generator.__getitem__(1)
sample_image = sample_image_batch[2,:,:,:]
#sample_mask=sample_mask_batch[2,:,:,:]
#display([sample_image, np.squeeze(sample_mask)])


#Model define
OUTPUT_CHANNELS=n_classes
model1 = segmentation_network(base_model_name,model_name, OUTPUT_CHANNELS, IMAGE_SIZE, ADD_FEAT)
w_loss = tf.keras.losses.CategoricalCrossentropy()


initial_learning_rate = 3*1e-5
k_iou_metric=tf.keras.metrics.MeanIoU(num_classes=n_classes)
opt=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, decay=1e-6)
opt2=tf.keras.optimizers.SGD(learning_rate=3*1e-5)
lr_poly=tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, 35000, end_learning_rate=0, power=0.9)
opt_poly=tf.keras.optimizers.SGD(learning_rate=lr_poly)
model=label_supervision(model1, opt, reg_consis)

if fire_db==1:
    checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
else:
    checkpint_folder = model_name+'-'+base_model_name+'-smoke-'+str(IMAGE_SIZE[0])

checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

input_image = sample_image_batch[1, ...]
[label_out, mask_out, mask_sig] = model.predict(input_image[np.newaxis, ...])
pred_mask=create_binary_mask(mask_out, cam_thre)
vis_pred=create_binary_mask(mask_sig, cam_thre)
display([input_image, np.squeeze(pred_mask[0]), np.squeeze(vis_pred[0])])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        input_image=sample_image_batch[2, ...]
        [label_out, mask_out, mask_sig] = model.predict(input_image[np.newaxis, ...])
        pred_mask = create_binary_mask(mask_out, cam_thre)
        pred_mask_vis = create_binary_mask(mask_sig, cam_thre)
        display([input_image, np.squeeze(pred_mask[0]), np.squeeze(pred_mask_vis[0])])



model.compile(optimizer=opt)

model_history=model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=[cp_callback])

