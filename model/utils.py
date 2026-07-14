import math
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, metrics, decomposition
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class OverlappingCrop:
    """
    Generate overlapping and evenly spaced crops.
    Returns list of PIL Images (see FiveCrop documentation)
    """
    def __init__(self, crop_size, stride, pad=True, mode='RGB'):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.stride = stride
        self.mode = mode
        self.pad = pad

    def __call__(self, data):
        image = np.array(data)
        h, w = image.shape[:2]
        new_h, new_w = self.crop_size

        # Pad image
        if self.pad and (h % new_h != 0 or w % new_w != 0):
            old_h = h
            old_w = w
            h = math.ceil(h / new_h) * new_h
            w = math.ceil(w / new_w) * new_w

            pad_h = h - old_h
            pad_w = w - old_w
            pad_h_top = math.floor(pad_h / 2)
            pad_h_bottom = math.ceil(pad_h / 2)
            pad_w_left = math.floor(pad_w / 2)
            pad_w_right = math.ceil(pad_w / 2)

            image = np.stack([np.pad(image[:,:,i], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant', constant_values=0) for i in range(image.shape[2])], axis=2)
            # image = np.stack([np.pad(image[:,:,i], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'symmetric') for i in range(image.shape[2])], axis=2)

        # Generate list of overlapping crops
        image_crops = [image[i:i+new_h, j:j+new_w, :] for i in range(0, h, self.stride)
                                                            for j in range(0, w, self.stride)]
        # Filter image crops
        image_crops = [Image.fromarray(crop.astype('uint8'), self.mode) for crop in image_crops if crop.shape[:2] == (new_h, new_w)]

        return image_crops

class OverlappingCropMultiChannel:
    """
    Generate overlapping and evenly spaced crops.
    Returns list of PIL Images (see FiveCrop documentation)
    """
    def __init__(self, crop_size, stride, pad=True, mode='RGB'):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.stride = stride
        self.mode = mode
        self.pad = pad

    def __call__(self, image):
        ch, h, w = image.size()
        new_h, new_w = self.crop_size

        # Pad image
        if self.pad and (h % new_h != 0 or w % new_w != 0):
            old_h = h
            old_w = w
            h = math.ceil(h / new_h) * new_h
            w = math.ceil(w / new_w) * new_w

            pad_h = h - old_h
            pad_w = w - old_w
            pad_h_top = math.floor(pad_h / 2)
            pad_h_bottom = math.ceil(pad_h / 2)
            pad_w_left = math.floor(pad_w / 2)
            pad_w_right = math.ceil(pad_w / 2)

            image = np.stack([np.pad(image[i,:,:], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant', constant_values=0) for i in range(ch)], axis=2)

        # Generate list of overlapping crops
        image_crops = [image[:, i:i+new_h, j:j+new_w] for i in range(0, h, self.stride)
                                                            for j in range(0, w, self.stride)]
        # Filter image crops
        image_crops = [crop for crop in image_crops if crop.size()[1:] == (new_h, new_w)]

        return image_crops


def plot_sample_batch(img, save_dir, save_name='sample_images.png'):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    fig = plt.figure(figsize = (30, 30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

def index(input_maps, input_choices):
    "Returns boolean list to index array"
    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

    return np.logical_and.reduce(idx_list_)

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def intersect_dicts(class_merge_dict, moa_dict):
    intrsct_dict = dict()
    for k, v in class_merge_dict.items():
        if v in moa_dict.keys():
            intrsct_dict[k] = moa_dict[v]

    return {**intrsct_dict, **moa_dict}

def convert_to_list(x):
    return [x_i.unsqueeze(0) for x_i in x]
