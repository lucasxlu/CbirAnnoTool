import json
import logging
import os
import pickle
import time

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from django.http import HttpResponse
from skimage import io
from skimage.color import gray2rgb
from torchvision import models
from torchvision.transforms import transforms

from cv.preprocess.densenet import DenseNet121

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_PORT = 'http://localhost:8000'
USE_GPU = True
LOSS = 2  # 0--SoftmaxLoss, 1--CenterLoss, 2--A-SoftmaxLoss
OUT_NUM = 391

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:[%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler('cbiranno.log'),
                        logging.StreamHandler()
                    ])


def build_faiss_index(nd_feats_array, mode):
    """
    build index on multi GPUs
    :param nd_feats_array:
    :param mode: 0: CPU; 1: GPU; 2: Multi-GPU
    :return:
    """
    d = nd_feats_array.shape[1]

    cpu_index = faiss.IndexFlatL2(d)  # build the index on CPU
    if mode == 0:
        logging.info("Is trained? >> {}".format(cpu_index.is_trained))
        cpu_index.add(nd_feats_array)  # add vectors to the index
        logging.info("Capacity of gallery: {}".format(cpu_index.ntotal))

        return cpu_index
    elif mode == 1:
        ngpus = faiss.get_num_gpus()
        logging.info("number of GPUs:", ngpus)
        res = faiss.StandardGpuResources()  # use a single GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(nd_feats_array)  # add vectors to the index
        logging.info("Capacity of gallery: {}".format(gpu_index.ntotal))

        return gpu_index
    elif mode == 2:
        multi_gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index on multi GPUs
        multi_gpu_index.add(nd_feats_array)  # add vectors to the index
        logging.info("Capacity of gallery: {}".format(multi_gpu_index.ntotal))

        return multi_gpu_index


def search(index, query_feat, topK):
    """
    search TopK results
    :param index:
    :param topK:
    :return:
    """
    xq = query_feat.astype('float32')
    D, I = index.search(xq, topK)  # actual search
    # print(D[:5])  # neighbors of the 5 first queries
    # print(I[:5])  # neighbors of the 5 first queries

    logging.debug(I)
    logging.debug(D)

    return I.ravel()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class ImageSearcher:
    def __init__(self, index_filename_pkl='cv/preprocess/idx_AdditiveFood.pkl',
                 feats_pkl='cv/preprocess/feats_AdditiveFood.pkl'):

        if LOSS == 0:
            # For Softmax Loss
            densenet121 = models.densenet121(pretrained=False)
            num_ftrs = densenet121.classifier.in_features
            densenet121.classifier = nn.Linear(num_ftrs, OUT_NUM)
        elif LOSS == 1 or LOSS == 2:
            # For CenterLoss and A-Softmax Loss
            densenet121 = DenseNet121(LOSS, OUT_NUM)

        state_dict = torch.load('/data/lucasxu/ModelZoo/DenseNet121_TissuePhysiology_Embedding_AngularLoss.pth')
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            densenet121.load_state_dict(new_state_dict)
        except:
            densenet121.load_state_dict(state_dict)
        device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        densenet121.eval()
        densenet121 = densenet121.to(device)

        logging.debug('loading gallery features')
        with open(feats_pkl, mode='rb') as f:
            nd_feats_array = pickle.load(f).astype('float32')
        logging.debug(nd_feats_array.shape)
        logging.debug('finish loading gallery\n[INFO] building index...')
        index = build_faiss_index(nd_feats_array, mode=0)
        logging.debug('finish building index...')

        with open(index_filename_pkl, mode='rb') as f:
            idx_filename = pickle.load(f)

        self.index = index
        self.idx_filename = idx_filename
        self.model = densenet121
        self.device = device

    def ext_deep_feat(self, img_filepath):
        """
        extract deep feature from an image filepath
        :param img_filepath:
        :return:
        """
        print(self.model)
        model_name = self.model.__class__.__name__

        self.model.eval()
        if model_name.startswith('DenseNet'):
            # print('extracting deep features of {}...'.format(model_name))

            with torch.no_grad():
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                image = io.imread(img_filepath)
                print('before color conversion: {}'.format(image.shape[-1]))
                if image.shape[-1] < 3:
                    image = gray2rgb(image)
                elif image.shape[-1] > 3:
                    # use skimage.color.rgba2rgb method brings bug!
                    # image = rgba2rgb(image)
                    image = image[:, :, 0:-1]
                print('after color conversion: {}'.format(image.shape[-1]))

                img = preprocess(Image.fromarray(image.astype(np.uint8)))
                img.unsqueeze_(0)
                img = img.to(self.device)

                inputs = img.to(self.device)

                if LOSS == 0:
                    feat = self.model.features(inputs)
                    feat = F.relu(feat, inplace=True)
                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
                elif LOSS == 1 or LOSS == 2:
                    feat = self.model.model.features(inputs)
                    feat = F.relu(feat, inplace=True)
                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

                    # print('feat size = {}'.format(feat.shape))
        feat = feat.to('cpu').detach().numpy()

        return feat / np.linalg.norm(feat)

    def search(self, query_img, topK=300):
        """
        search TopK results:
        :param topK:
        :return:
        """
        query_feat = self.ext_deep_feat(query_img)
        xq = query_feat.astype('float32')
        D, I = self.index.search(xq, topK)  # actual search
        # print(D[:5])  # neighbors of the 5 first queries
        # print(I[:5])  # neighbors of the 5 first queries

        logging.debug(I)
        logging.debug(D)

        returned_indices = I.ravel()

        return [os.path.basename(self.idx_filename[idx]) for idx in returned_indices]


image_searcher = ImageSearcher()


def upload_and_search(request):
    """
    upload and search image
    :param request:
    :return:
    """
    image_dir = 'cv/static/CBIRUpload'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result = {}

    if request.method == "POST":
        image = request.FILES.get("image", None)
        if not image:
            result['code'] = 1
            result['msg'] = 'Invalid Image'
            result['data'] = None

            json_result = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_result)
        else:
            destination = open(os.path.join(image_dir, image.name), 'wb+')
            for chunk in image.chunks():
                destination.write(chunk)
            destination.close()

            tik = time.time()

            res = image_searcher.search(os.path.join(image_dir, image.name))

            if res is not None:
                result['code'] = 0
                result['msg'] = 'success'
                result['data'] = res
                result['elapse'] = round(time.time() - tik, 2)
            else:
                result['code'] = 3
                result['msg'] = 'None item is retrieved'
                result['data'] = None
                result['elapse'] = round(time.time() - tik, 2)

            json_str = json.dumps(result, ensure_ascii=False)

            return HttpResponse(json_str)
    else:
        result['code'] = 2
        result['msg'] = 'Invalid HTTP Method'
        result['data'] = None

        json_result = json.dumps(result, ensure_ascii=False)

        return HttpResponse(json_result)
