import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from utils import labels2onehot_tf

def binary_numpy_iou(y_true, y_pred):
    num_classes = 1
    y_true_bin=np.ones_like(y_true)
    y_pred_bin=np.ones_like(y_pred)
    y_true_bin[y_true<.5]=0
    y_pred_bin[y_pred<.5]=0
    y_true=y_true_bin
    y_pred=y_pred_bin
    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    iou = np.mean(iou)
    dice = np.mean(dice)
    return iou

def my_binary_numpy_iou(y_true, y_pred, thre):
    num_classes = 1
    y_true_bin=np.ones_like(y_true, dtype=float)
    y_pred_bin=np.ones_like(y_pred, dtype=float)
    y_true_bin[y_true<.5]=0.0
    y_pred_bin[y_pred<.5]=0.0
    y_true=y_true_bin
    y_pred=y_pred_bin
    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.multiply(y_pred, y_true), axis=axes)
    union = np.sum(y_true, axis=axes)+np.sum(y_pred, axis=axes)-intersection
    smooth = .001
    iou = (intersection+ smooth) / (union + smooth)

    iou = np.mean(iou)
    return iou

def numpy_iou(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    y_pred = np.array([np.argmax(y_pred, axis=-1) == i for i in range(num_classes)]).transpose(1, 2, 3, 0)
    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    iou = np.mean(iou)
    dice = np.mean(dice)
    return iou

def my_accuarcy_metric(y_true, y_pred):
    intesct1=tf.math.multiply(y_true,y_pred)
    intesct1=tf.reduce_sum(intesct1, axis=-1)
    intersect=tf.reduce_sum(intesct1, axis=[1,2])
    shape_y=tf.shape(y_true)
    sizy_y=tf.multiply(shape_y[1], shape_y[2])
    sizy_y=tf.dtypes.cast(sizy_y, tf.int64)
    acc_by_sample=tf.math.divide(intersect, sizy_y)
    return tf.reduce_mean(acc_by_sample)

def my_miou_metric(y_true, y_pred):
    ind_true=tf.keras.backend.argmax(y_true)
    ind_pred = tf.keras.backend.argmax(y_pred)
    dpth=tf.constant(21, dtype=np.int32)
    y_true=tf.one_hot(ind_true, dpth)
    y_pred = tf.one_hot(ind_pred, dpth)
    union1=tf.math.add(y_true, y_pred)
    onemat=tf.ones_like(union1)
    union2 = tf.where(tf.equal(2.0, union1), onemat, union1)
   # union2=union1
    intesct1=tf.math.multiply(y_true,y_pred)
    intersectv=tf.reduce_sum(intesct1, axis=[1,2])
    union=tf.reduce_sum(union2, axis=[1,2])
    num_nz=tf.math.count_nonzero(union, axis=1, dtype=tf.float32)
    onevec = tf.ones_like(union)
    unionv = tf.where(tf.equal(0.0, union), onevec, union)
    point_iou=tf.keras.backend.sum(tf.math.divide(intersectv,unionv), axis=1)/num_nz
    return tf.math.reduce_mean(point_iou)


def other_accuracy(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = np.sum(y_true * y_pred, axis=-1)
        intersection=np.sum(intersection, axis=(1, 2))
        union = y_true.shape[1]*y_true.shape[2]
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return np.mean(x)

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def others_iou(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def me_others_iou(y_true, y_pred, smooth=0.0001,re_lst_index=0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    intersection=tf.cast(intersection, dtype=tf.float64)
    union=tf.cast(union, dtype=tf.float64)
    num_nz = tf.math.count_nonzero(K.sum(y_true, [1, 2]), axis=1, dtype=tf.float64)
    if re_lst_index:
        iouc = K.sum((intersection[:,:-1])/ (union[:,:-1] + smooth), axis=1) / (num_nz-1)
    else:
        iouc=K.sum((intersection) / (union + smooth), axis=1)/num_nz
    iou=K.mean(iouc)
    return iou

def general_iou(y_true, y_pred, smooth=0.0001,re_lst_index=0):

    y_true_label=tf.math.reduce_max(y_true, axis=-1, keepdims=True)
    y_pred_label=tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
    y_true_max=tf.where(tf.equal(y_true_label, tf.zeros_like(y_true_label)),tf.ones_like(y_true_label),y_true_label)
    y_pred_max=tf.where(tf.equal(y_pred_label, tf.zeros_like(y_pred_label)),tf.ones_like(y_pred_label),y_pred_label)
    y_true_onehot=tf.cast(tf.math.equal(y_true, y_true_max), dtype=tf.int32)
    y_pred_onehot=tf.cast(tf.math.equal(y_pred, y_pred_max), dtype=tf.int32)
    y_true=y_true_onehot
    y_pred=y_pred_onehot
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    intersection=tf.cast(intersection, dtype=tf.float64)
    union=tf.cast(union, dtype=tf.float64)
    num_nz = tf.math.count_nonzero(K.sum(y_true, [1, 2]), axis=1, dtype=tf.float64)
    if re_lst_index:
        iouc = K.sum((intersection[:,:-1])/ (union[:,:-1] + smooth), axis=1) / (num_nz-1)
    else:
        iouc=K.sum((intersection) / (union + smooth), axis=1)/num_nz
    iou=K.mean(iouc)
    return iou


def binary_miou_metric(y_true, y_pred, smooth=.0001):
    y_pred=tf.where(tf.math.less(y_pred, tf.constant([.5])), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    y_pred=tf.expand_dims(y_pred, axis=-1)
    y_true=tf.expand_dims(y_true, axis=-1)
    y_pred=tf.concat([1-y_pred, y_pred], 3)
    y_true=tf.concat([1-y_true, y_true], 3)
    y_pred=tf.cast(y_pred, dtype=tf.float64)
    y_true = tf.cast(y_true, dtype=tf.float64)
    num_nz = tf.math.count_nonzero(tf.reduce_sum(y_true, [1, 2]), axis=1, dtype=tf.float64)
    intersection = tf.reduce_sum(tf.math.abs(tf.multiply(y_true, y_pred)), axis=[1, 2])
    union = tf.reduce_sum(y_true, [1, 2]) + tf.reduce_sum(y_pred, [1, 2]) - intersection
    return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.divide(intersection, tf.add(union, smooth)), axis=1),num_nz))


def binary_miou_metric2(y_true, y_pred):
    z_true=tf.zeros_like(y_true)
    o_true = tf.ones_like(y_true)
    z_pred = tf.zeros_like(y_pred)
    o_pred=tf.ones_like(y_pred)
    y_true=tf.where(tf.math.less(y_true, tf.constant([.5])), z_true, o_true)
    y_pred=tf.where(tf.math.less(y_pred, tf.constant([.5])), z_pred, o_pred)

    union1=tf.math.add(y_true, y_pred)
    onemat=tf.ones_like(union1)
    union2 = tf.where(tf.equal(2.0, union1), onemat, union1)
   # union2=union1
    intesct1=tf.math.multiply(y_true,y_pred)
    intersectv=tf.reduce_sum(intesct1, axis=[1,2])
    union=tf.reduce_sum(union2, axis=[1,2])
    onevec = tf.ones_like(union)
    unionv = tf.where(tf.equal(0.0, union), onevec, union)
    point_iou=tf.keras.backend.sum(tf.math.divide(intersectv,unionv), axis=1)
    return tf.math.reduce_mean(point_iou)


def general_iou_ignore(y_true, y_pred, smooth=0.0001,re_lst_index=0):

    y_true_label=tf.math.reduce_max(y_true, axis=-1, keepdims=True)
    y_pred_label=tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
    y_true_max=tf.where(tf.equal(y_true_label, tf.zeros_like(y_true_label)),tf.ones_like(y_true_label),y_true_label)
    y_pred_max=tf.where(tf.equal(y_pred_label, tf.zeros_like(y_pred_label)),tf.ones_like(y_pred_label),y_pred_label)
    y_true_onehot=tf.cast(tf.math.equal(y_true, y_true_max), dtype=tf.int32)
    y_pred_onehot=tf.cast(tf.math.equal(y_pred, y_pred_max), dtype=tf.int32)
    y_true=y_true_onehot
    y_pred=y_pred_onehot
    y_true_sum=tf.expand_dims(tf.reduce_sum(y_true, axis=-1), axis=-1)
    y_pred=tf.where(tf.equal(y_true_sum, tf.zeros_like(y_true_sum)), tf.zeros_like(y_pred), y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    intersection=tf.cast(intersection, dtype=tf.float64)
    union=tf.cast(union, dtype=tf.float64)
    num_nz = tf.math.count_nonzero(K.sum(y_true, [1, 2]), axis=1, dtype=tf.float64)
    if re_lst_index:
        iouc = K.sum((intersection[:,:-1])/ (union[:,:-1] + smooth), axis=1) / (num_nz-1)
    else:
        iouc=K.sum((intersection) / (union + smooth), axis=1)/num_nz
    iou=K.mean(iouc)
    return iou
a=np.array([[[1,1,1],[0,0,1]], [[1,0,0],[1,1,1]]])
b=np.array([[[1,1,1],[0,0,1]], [[1,0,0],[1,1,1]]])
v_ioub=binary_numpy_iou(a,b)
my_ioub=my_binary_numpy_iou(a,b,.5)
print(v_ioub, my_ioub)
#acc_val=general_iou(a,b)
#print(acc_val)
