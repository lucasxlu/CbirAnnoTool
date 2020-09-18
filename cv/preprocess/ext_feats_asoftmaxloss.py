import os
import sys
import pickle
import math
import time

from PIL import Image
import numpy as np
from skimage import io

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import gray2rgb, rgba2rgb
from torch.nn import Parameter
from torchvision import models
from torchvision.transforms import transforms

from densenet import DenseNet121


USE_GPU = True
CLS_NAME = 'AdditiveFood'


def load_model_with_weights(model, model_path):
    """
    load model with pretrained checkpoint
    :param model:
    :param model_path
    :return:
    """
    print(model)
    model = model.float()
    model_name = model.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if model_path is not None and model_name != "":
        model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.device_count() > 1:
        model = model.module

    return model


def ext_deep_feat(model_with_weights, img_filepath):
    """
    extract deep feature from an image filepath
    :param model_with_weights:
    :param img_filepath:
    :return:
    """
    model_name = model_with_weights.__class__.__name__
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
    model_with_weights = model_with_weights.to(device)
    model_with_weights.eval()
    if model_name.startswith('DenseNet'):
        with torch.no_grad():
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = io.imread(img_filepath)
            if len(list(image.shape)) < 3:
                image = gray2rgb(image)
            elif len(list(image.shape)) > 3:
                image = rgba2rgb(image)

            img = preprocess(Image.fromarray(image.astype(np.uint8)))
            img.unsqueeze_(0)

            inputs = img.to(device)

            feat = model_with_weights.embedding(model_with_weights.features(inputs))

            # print('feat size = {}'.format(feat.shape))

    feat = feat.to('cpu').detach().numpy()
    feat = feat / np.linalg.norm(feat)

    return feat


def ext_feats_in_dir(model_with_weights, gallery_img_root):
    """
    extract deep features in a directory
    :param model_with_weights:
    :param gallery_img_root:
    :return:
    """
    print(model_with_weights)
    model_with_weights.eval()
    print('[INFO] start extracting features')
    idx_filename = {}
    feats = []
    idx = 0
    capacity_of_gallery = len(os.listdir(gallery_img_root))

    for img_f in sorted(os.listdir(gallery_img_root)):
        tik = time.time()
        feat = ext_deep_feat(model_with_weights, os.path.join(gallery_img_root, img_f))
        tok = time.time()
        print('[INFO] {0}/{1} extracting deep features, feat size = {2}, latency = {3}'.format(idx, capacity_of_gallery, feat.shape, tok - tik))

        idx_filename[idx] = '{}'.format(img_f)
        feats.append(feat.ravel().tolist())
        idx += 1
    print('[INFO] finish extracting features')

    with open('feats_%s.pkl'%CLS_NAME, mode='wb') as f:
        pickle.dump(np.array(feats).astype('float32'), f)

    with open('idx_%s.pkl'%CLS_NAME, mode='wb') as f:
        pickle.dump(idx_filename, f)


if __name__ == '__main__':
    densenet121 = DenseNet121(num_cls=362)

    state_dict = torch.load('/data/lucasxu/ModelZoo/DenseNet121_{}_Embedding_ASoftmaxLoss.pth'.format(CLS_NAME))
    try:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        densenet121.load_state_dict(new_state_dict)
    except:
        densenet121.load_state_dict(state_dict)

    ext_feats_in_dir(densenet121, '/data/lucasxu/Dataset/{}Crops/{}'.format(CLS_NAME, CLS_NAME))
