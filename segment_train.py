import numpy as np
import tensorflow as tf
from PIL import Image
from models import segmentation_network
from my_metrics import binary_miou_metric, binary_numpy_iou, my_binary_numpy_iou
from generators import fire_image_generator
from utils import create_tf_binary_mask, binary_labels_onehot, binary_labels_onehot_tf, display, tf_compute_weights
import os


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
            pred_label_rot, pred_mask_rot, pred_jnk = self.model(sample_image_batch_rot, training=True)
            pred_mask_360 = tf.image.rot90(pred_mask_rot, 4 - k_rot)

            # pred_mask_bg = create_mask_bg(pred_mask)

            label_loss = self.b_ce_loss(labels_batch, pred_label)
            segment_loss = self.b_ce_loss(sample_mask_batch, pred_mask)
            sup_loss=tf.keras.losses.mse(pred_mask, pred_mask_sig)
            consistency_loss = tf.keras.losses.mse(pred_mask, pred_mask_360)
            overall_loss = label_loss+self.reg*sup_loss+0.01*consistency_loss

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(overall_loss, trainable_vars)
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        v_label_metric = self.label_metric.update_state(labels_batch, pred_label)
        pred_mask=create_tf_binary_mask(pred_mask, .4)
        v_segment_iou = self.miou_metric(sample_mask_batch, pred_mask)

        return {"Label_loss": label_loss, "Segmentation_loss": segment_loss, "label_acc": self.label_metric.result(),
                "Segment_IOU": v_segment_iou}





shuffle=1
cam_thre=.6
cam_thre2=.45
mid_lev_train=1
BATCH_SIZE = 12
IMAGE_SIZE=(256,256)
ANY_SIZE=1
model_name='fcn-16'
base_model_name='vgg'
MASK_SIZE=IMAGE_SIZE
ADD_FEAT=3
n_classes=1
EPOCHS = 50
test_img_dir='../dataset/test_weakly_2/rgb/rgb/'
test_mask_dir='../dataset/test_weakly_2/masked/masked/'
train_img_dir='../dataset/Weakly_fire/images/'
train_mask_dir='../dataset/Weakly_fire/masks/'
normalize=1

train_ids = next(os.walk(train_img_dir))[2]
test_ids = next(os.walk(test_img_dir))[2]
train_generator=fire_image_generator(train_img_dir, train_mask_dir, train_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, shuffle)
test_generator=fire_image_generator(test_img_dir, test_mask_dir, test_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, shuffle)


OUTPUT_CHANNELS=n_classes
model1 = segmentation_network(base_model_name,model_name, OUTPUT_CHANNELS, IMAGE_SIZE,ADD_FEAT)
model=label_supervision(model1)
model_seg = segmentation_network('vgg','deeplab', OUTPUT_CHANNELS, IMAGE_SIZE,ADD_FEAT)

checkpint_folder_seg = model_name+'-'+base_model_name+'-'+str(mid_lev_train)+str(IMAGE_SIZE[0])
checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-0035.ckpt'
model.load_weights(checkpoint_path)

num_train_batches=train_generator.__len__()
num_test_batches=test_generator.__len__()
#print('Number of samples is:{}'. format(test_generator.num_of_samples()))
#num_epochs=6


label_acc_metric = tf.keras.metrics.BinaryAccuracy()
test_acc_metric = tf.keras.metrics.BinaryAccuracy()
loss_func=tf.keras.losses.BinaryCrossentropy()
opt=tf.keras.optimizers.Adam(learning_rate=5*1e-5, decay=1e-6)

lmd=0
for j in range(EPOCHS):
    iou_v = np.zeros(num_train_batches)
    iou_v2 = np.zeros(num_train_batches)
    iou_v_test=np.zeros(num_test_batches)
    label_acc_metric.reset_states()
    test_acc_metric.reset_states()
    if j>0:
        lmd=1
    if j%10 ==0 and j>15:
        checkpoint_path_seg = 'checkpoints/' + checkpint_folder_seg + '/cp-any448-{j:04d}.ckpt'
        model_seg.save_weights(checkpoint_path_seg)

    for i in range(num_train_batches):
        sample_image_batch, [sample_label_batch, sample_mask_batch] = train_generator.__getitem__(i)
        pred_label, pred_mask, pred_sig = model(sample_image_batch, training=False)
        sample_label_onehot=binary_labels_onehot_tf(pred_label)
        #pred_mask_w=tf_compute_weights(pred_mask, .6, .95)
        #pred_mask_w_mid = tf_compute_weights(pred_mask, .35, .7)
        pred_mask=create_tf_binary_mask(pred_mask, cam_thre)
        #pred_sig = binary_labels_onehot_tf(pred_sig)
        pred_sig = create_tf_binary_mask(pred_sig, cam_thre2)
        pred_mask=tf.multiply(sample_label_onehot[...,tf.newaxis,tf.newaxis], pred_mask)
        pred_sig=tf.multiply(sample_label_onehot[...,tf.newaxis,tf.newaxis], pred_sig)
        #display([sample_image_batch[2], np.squeeze(pred_mask[2])])
        pr_lbl,pr_logits,pr_seg_masks=model_seg(sample_image_batch, training=False)
        pr_seg_masks = binary_labels_onehot_tf(pr_seg_masks)
        pr_seg_masks = tf.multiply(sample_label_onehot[..., tf.newaxis, tf.newaxis], pr_seg_masks)
        #pr_logits_w = tf_compute_weights(pr_logits, .6, .85)



        with tf.GradientTape() as tape:
            p_lbl,p_logits, seg_pred_mask=model_seg(sample_image_batch, training=True)
            loss_reg=loss_func(pr_seg_masks,seg_pred_mask)
            if mid_lev_train==1:
                loss_val = loss_func(pred_sig, seg_pred_mask)+lmd*loss_reg
            else:
                loss_val=loss_func(pred_mask,seg_pred_mask)+lmd*loss_reg

        trainable_vars=model_seg.trainable_variables
        grads=tape.gradient(loss_val, trainable_vars)
        opt.apply_gradients(zip(grads, trainable_vars))
        label_acc_metric.update_state(sample_mask_batch, seg_pred_mask)
        iou_v[i]=my_binary_numpy_iou(sample_mask_batch, seg_pred_mask.numpy(), .5)

        if i % 10 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (i, float(loss_val))
            )
            print('ACC is {}, and the IOU is {}'.format(label_acc_metric.result().numpy(), np.mean(iou_v[:i+1])))

    for k in range(num_test_batches):
        sample_image_batch, [sample_label_batch, sample_mask_batch] = test_generator.__getitem__(k)
        pred_label, pred_mask, pred_sig = model_seg(sample_image_batch, training=False)

        #sample_label_onehot = binary_labels_onehot_tf(sample_label_batch)
        pred_sig = binary_labels_onehot_tf(pred_sig)
        pred_sig = tf.multiply(sample_label_batch[..., tf.newaxis, tf.newaxis, tf.newaxis], pred_sig)
        if j==12:
            display([np.squeeze(sample_mask_batch[2]), np.squeeze(pred_sig[2])])
        test_acc_metric.update_state(sample_mask_batch, pred_sig)
        iou_v_test[k] = my_binary_numpy_iou(sample_mask_batch, pred_sig.numpy(), .5)
    print('======> End of epoch {}, The validation ACC {} and IOU {}'.format(j, test_acc_metric.result().numpy(), np.mean(iou_v_test)))
