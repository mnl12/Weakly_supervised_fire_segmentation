import numpy as np
from PIL import Image

def colors2labels(im, cmap, one_hot=False):
    if one_hot:
        labels = np.zeros((*im.shape[:-1], len(cmap)), dtype='uint8')
        for i, color in enumerate(cmap):
            labels[:, :, i] = np.all(im == color, axis=2).astype('uint8')
    else:
        labels = np.zeros(im.shape[:-1], dtype='uint8')
        for i, color in enumerate(cmap):
            labels += i * np.all(im == color, axis=2).astype(dtype='uint8')
    return labels


def labels2colors(labels, cmap):
    labels_colored = np.zeros((*labels.shape, 3), dtype='uint8')
    for label in np.unique(labels):
        label_mask = labels == label
        label_mask = np.dot(np.expand_dims(label_mask, axis=2), np.array(cmap[label]).reshape((1, -1)))
        labels_colored += label_mask.astype('uint8')
    return labels_colored



def onehot2mask(onehotmask):
    """converts one-hot vector into image"""
    labeled_image=np.argmax(onehotmask, axis=2)
    return labeled_image


def _bitget(byteval, idx):
    return (byteval & (1 << idx)) != 0

def color_map1(n_classes=256, normalized=False):

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n_classes, 3), dtype=dtype)
    for i in range(n_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (_bitget(c, 0) << 7 - j)
            g = g | (_bitget(c, 1) << 7 - j)
            b = b | (_bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def color_map(n_classes):
    cmap = color_map1(256)
    cmap = np.vstack([cmap[:n_classes], cmap[-1].reshape(1, 3)])
    return cmap

def labels2onehot(img_batch, num_channels):
    img_batch=img_batch[..., np.newaxis]
    onehot = np.zeros((*img_batch.shape[:-1], num_channels), dtype='uint8')
    for i in range(num_channels):
        onehot[:, :, i] = np.all(img_batch == i, axis=2).astype('uint8')
    return onehot

#cmap=color_map(21)
#sample_mask=np.asarray(Image.open('../dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000123.png').convert('RGB'))
#label_image=colors2labels(sample_mask, cmap, False)
#out_mask=labels2onehot(label_image, 22)
#print(out_mask)
