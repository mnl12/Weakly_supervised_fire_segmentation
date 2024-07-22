import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors, labels2onehot


def sc_inv_avg_poll(inputs, thre):
    max_input = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
    input_thre = tf.where(tf.less(inputs, tf.multiply(tf.multiply(thre, max_input), tf.ones_like(inputs))),
                          tf.zeros_like(inputs), inputs)
    num_nonzeros = tf.math.count_nonzero(input_thre, axis=[1, 2], dtype=tf.float32)
    num_nonzeros=tf.math.add(num_nonzeros, tf.ones_like(num_nonzeros))
    out = tf.math.divide(tf.reduce_sum(input_thre, axis=[1, 2]), num_nonzeros)
    out = tf.math.add(out, tf.multiply(.01, tf.ones_like(out)))
    return out

def labels2onehot(img_batch, num_channels):
    img_batch=img_batch[..., np.newaxis]
    onehot = np.zeros((*img_batch.shape[:-1], num_channels), dtype='float64')
    for i in range(num_channels):
        onehot[:, :, :, i] = np.all(img_batch == i, axis=3).astype('float64')
    return onehot

def labels2onehot_tf(input, num_channels):
    return tf.numpy_function(labels2onehot, [input,num_channels], tf.float64)


def binary_labels_onehot(input):
    onehot=np.ones_like(input)
    onehot[input<.5]=0
    return onehot

def binary_labels_onehot_tf(input):
    return tf.numpy_function(binary_labels_onehot, [input], tf.float64)


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

def create_mask(pred_mask):
    pred_mask_1=tf.argmax(pred_mask, axis=-1)
    pred_mask_c=np.empty((0,pred_mask_1.shape[1],pred_mask_1.shape[2], pred_mask.shape[-1]))
    for pred_mask_i in pred_mask_1:
        pred_mask_i_c=labels2colors(pred_mask_i, c_map)
        pred_mask_i_c=colors2labels(pred_mask_i_c, c_map, True)
        pred_mask_c=np.append(pred_mask_c, pred_mask_i_c[tf.newaxis, ...], axis=0)
    return pred_mask_c

def create_mask2(pred_mask,c_map):
    pred_mask1=tf.argmax(pred_mask, axis=-1)
    pred_mask=pred_mask1[0]
    pred_mask=labels2colors(pred_mask, c_map)
    return pred_mask

def create_mask_tr(pred_mask_batch, c_map, thre):
    mask_thre_batch=(pred_mask_batch>thre)*pred_mask_batch
    mask_label_batch=[np.argmax(mask_i, axis=-1) for mask_i in mask_thre_batch]
    return np.array([labels2colors(imask, c_map) for imask in mask_label_batch], dtype='uint8')

def create_binary_mask(pred_mask_b, thre):
    max_val=np.max(pred_mask_b, axis=(1,2), keepdims=True)
    return np.where(pred_mask_b < thre*max_val, np.zeros_like(pred_mask_b), np.ones_like(pred_mask_b))

def create_tf_binary_mask(input, thre):
    y=tf.numpy_function(create_binary_mask, [input, thre], tf.float32)
    return y

def create_mask_batch(mask_batch):
    return np.array([labels2onehot(np.argmax(mask_i, axis=-1), 22) for mask_i in mask_batch], dtype='uint8')

def create_mask_tf_wrapper(input):
    y = tf.numpy_function(create_mask_batch, [input], tf.uint8)
    return y



def validate_results():
    miou_v=np.zeros(TEST_LENGTH//BATCH_SIZE)
    for i in range(TEST_LENGTH//BATCH_SIZE):
        sample_image_batch, sample_mask_batch=test_generator.__getitem__(i)
        pred_mask_batch = model.predict(sample_image_batch)
        pred_mask_bin_batch = create_mask(pred_mask_batch)
        miou_val = me_others_iou(sample_mask_batch, pred_mask_bin_batch)
        miou_v[i]=miou_val

    print('IOU value:', np.mean(miou_v))

def compute_weights(inputs, thre_min, thre_max):
    max_val = np.max(inputs, axis=(1, 2), keepdims=True)
    cond=np.logical_or(inputs > (thre_max * max_val) , inputs<(thre_min * max_val))
    return np.where(cond, np.ones_like(inputs), np.zeros_like(inputs))

def tf_compute_weights(input, thre_min, thre_max):
    y=tf.numpy_function(compute_weights, [input, thre_min, thre_max], tf.float32)
    return y

#a=tf.constant([[[[0.1,0.8,0.3],[.1,0.2,0.3],[.1,0.5,0.4]],[[0.6,0.3,.1],[.1,0.3,0.6],[.1,0.4,0.6]]], [[[0.5,0.7,.1],[.5,0.3,0.6],[.4,0.3,0.6]],[[0.8,0.5,.3],[.2,0.5,0.7],[.1,0.4,0.7]]]])
#b=sc_inv_avg_poll(a, .5)
#b=tf.linalg.normalize(a)
#print(b)
#b=np.array([[[[0,1,0],[0,1,0],[0,1,0]],[[0,0,1],[1,0,0],[0,1,0]]], [[[0,1,0],[0,1,0],[0,0,1]],[[0,1,0],[1,0,0],[0,1,0]]]])
