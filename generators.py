import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import os
import random
from PIL import Image
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2onehot

class fire_image_generator (tf.keras.utils.Sequence):
    def __init__(self, image_path, mask_path, ids, batch_size, image_size, mask_size, normalization, shuffle_ind):
        self.indexes=ids
        self.image_path=image_path
        self.mask_path=mask_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.mask_size=mask_size
        self.batch_size=batch_size
        self.normalization=normalization
        self.shuffle=shuffle_ind


    def __len__(self):
        return int(len(self.indexes)/float(self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        random.shuffle(self.indexes)



    def get_labels(self, masks):
        labels=[]
        for mask in masks:
            if np.sum(mask)>0:
                label=1
            else:
                label=0
            labels.append(label)
        return np.array(labels)


    def __getitem__(self, index):
        indexes= self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images=[np.asarray(Image.open(self.image_path+k).convert('RGB').resize(self.image_size), dtype=float) for k in indexes]
        masks=[np.asarray(Image.open(self.mask_path+k).convert('L').resize(self.mask_size), dtype=float) for k in indexes]
        if self.normalization:
            images=np.array(images)/255.0
            masks=np.array(masks)/255.0
            masks=masks[..., tf.newaxis]
        labels=self.get_labels(masks)
        return images, [labels, masks]


class smoke_image_generator(tf.keras.utils.Sequence):
    def __init__(self, test_img_dir,test_mask_dir,train_smoke_dir,train_nosmoke_dir, batch_size, image_size, mask_size, normalization, shuffle_ind, train_ind):

        self.test_img_dir=test_img_dir
        self.test_mask_dir=test_mask_dir
        self.train_smoke_dir=train_smoke_dir
        self.train_nosmoke_dir=train_nosmoke_dir
        self.batch_size=batch_size
        self.image_size=image_size
        self.mask_size=mask_size
        self.batch_size=batch_size
        self.normalization=normalization
        self.shuffle=shuffle_ind
        self.train_ind=train_ind
        if train_ind:
            self.filename_s=next(os.walk(self.train_smoke_dir))[2]
            self.ids_s=[train_smoke_dir + fn for fn in self.filename_s]
            self.labels_s=np.ones(len(self.ids_s), dtype=int)
            self.filename_ns=next(os.walk(self.train_nosmoke_dir))[2]
            self.ids_ns = [train_nosmoke_dir + fn for fn in self.filename_ns]
            self.labels_ns=np.zeros(len(self.ids_ns), dtype=int)
            self.ids=np.concatenate((self.ids_s,self.ids_ns))
            self.labels=np.concatenate((self.labels_s,self.labels_ns))
            self.data_label=np.asarray([self.ids, self.labels]).T
        else:
            self.ids = next(os.walk(self.test_img_dir))[2]

    def __len__(self):
        return int(len(self.ids) / float(self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.train_ind:
            np.random.shuffle(self.data_label)

    def get_labels(self, masks):
        labels=[]
        for mask in masks:
            if np.sum(mask)>0:
                label=1
            else:
                label=0
            labels.append(label)
        return np.array(labels)

    def __getitem__(self, index):
        if self.train_ind:
            indexes = self.data_label[index * self.batch_size:(index + 1) * self.batch_size, 0]
            labels=self.data_label[index * self.batch_size:(index + 1) * self.batch_size, 1]
            labels=np.asarray(labels, dtype=float)
            images = [np.asarray(Image.open(k).convert('RGB').resize(self.image_size), dtype=float)
                      for k in indexes]
            images=np.asarray(images)
            masks=10.0*np.ones(self.image_size)

        else:

            indexes= self.ids[index*self.batch_size:(index+1)*self.batch_size]
            images = np.asarray([np.asarray(Image.open(self.test_img_dir + k).convert('RGB').resize(self.image_size), dtype=float)
                      for k in indexes])
            masks = np.asarray([np.asarray(Image.open(self.test_mask_dir + k).convert('L').resize(self.mask_size), dtype=float) for k
                     in indexes])
            masks[masks<254]=0
            if self.normalization:
                images = np.array(images) / 255.0

                masks = np.array(masks) / 255.0
                masks = masks[..., tf.newaxis]
            labels = self.get_labels(masks)
        return images, [labels, masks]







class pascal_voc_generator (tf.keras.utils.Sequence):
    def __init__(self, image_path, mask_path, ids, batch_size, image_size, mask_size, normalization, n_classes, any_size, shuffle, augment, extra):
        self.indexes=ids
        self.image_path=image_path
        self.mask_path=mask_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.mask_size=mask_size
        self.batch_size=batch_size
        self.normalization=normalization
        self.shuffle=shuffle
        self.n_classes=n_classes
        self.cmap=self.color_map_c()
        self.any_size=any_size
        self.aug=augment
        self.extra=extra

    def __len__(self):
        return int(len(self.indexes)/float(self.batch_size))

    def color_map_c(self):
        cmap256=color_map(256)
        cmap=np.vstack([cmap256[:self.n_classes], cmap256[-1].reshape(1, 3)])
        return cmap

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        if self.shuffle:
            random.shuffle(self.indexes)


    def get_labels(self, masks):
        label=np.sum(masks, axis=(1, 2))
        label_o=np.zeros_like(label)
        label_o[label>0]=1
        label_o[:,-1]=0
        return label_o

    def construct_image_batch(self, image_group, BATCH_SIZE):
        # get the max image shape
        #max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        max_shape=(*self.image_size, image_group[1].shape[2])

        # construct an image batch object
        image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            min_shape = tuple(min(image.shape[d], self.image_size[d]) for d in range(2))
            image_batch[image_index, :min_shape[0], :min_shape[1]] = image[:min_shape[0], :min_shape[1]]

        return image_batch

    def construct_mask_batch(self, image_group, BATCH_SIZE):
        # get the max image shape
        max_shape=(*self.image_size, image_group[1].shape[2])

        # construct an image batch object
        image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')
        #image_batch[:, :, :, self.n_classes] = 1


        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            min_shape = tuple(min(image.shape[d], self.image_size[d]) for d in range(2))
            image_batch[image_index, :min_shape[0], :min_shape[1]] = image[:min_shape[0], :min_shape[1]]
            image_batch[:, :, :, self.n_classes] = 0
            #Ignore Background
           # if np.random.choice(2,1, p=[.01,.99])==0:
           # image_batch[:, :, :, 0] = 0
            image_weights = np.sum(image_batch, axis=-1)


        return image_batch


    def __getitem__(self, index):
        if self.any_size==1:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            if self.aug==1:
                transform_ops = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
                rand_op = random.choice(transform_ops)
                rot_deg=np.random.randint(-20,20)
                images_orig=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB'), dtype=float) for k in indexes], self.batch_size)
                images_aug=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB').rotate(rot_deg), dtype=float) for k in indexes], self.batch_size)
                masks_onehot_orig = self.construct_image_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB')), self.cmap, True) for k in indexes], self.batch_size)
                masks_onehot_aug = self.construct_mask_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').rotate(rot_deg)), self.cmap, True) for k in indexes], self.batch_size)
                images = images_aug
                masks_onehot = masks_onehot_aug
                weights_masks = np.sum(masks_onehot, axis=-1, dtype=int)

            else:
                images=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png', 'jpg')).convert('RGB'), dtype=float) for k in indexes], self.batch_size)
                if self.extra == 0:
                    masks_onehot = self.construct_mask_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB')), self.cmap, True) for k in indexes], self.batch_size)
                    #masks=[colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').resize(self.image_size, Image.NEAREST)), self.cmap, False) for k in indexes]
                else:
                    masks_onehot = self.construct_mask_batch([labels2onehot(np.array(loadmat(self.mask_path+k.replace('png', 'mat'))['GTcls'][0,0][1]), self.n_classes+1) for k in indexes], self.batch_size)
                weights_masks = np.sum(masks_onehot, axis=-1, dtype=int)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            images=[np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB').resize(self.image_size), dtype=float) for k in indexes]
            masks_onehot = [colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').resize(self.image_size, Image.NEAREST)), self.cmap, True) for k in indexes]
            weights_masks=np.sum(masks_onehot, axis=-1, dtype=int)



        if self.normalization:
            images=np.array(images)/255.0
            #images=tf.keras.applications.vgg16.preprocess_input(images)
            #masks=np.array(masks, dtype=int)
            masks_onehot = np.array(masks_onehot, dtype=int)


        labels=self.get_labels(masks_onehot)
        label_weights=np.ones((labels.shape[0], 1), dtype=float)
        label_weights[0,:]=0
        return images, [labels, masks_onehot], [label_weights, weights_masks]
