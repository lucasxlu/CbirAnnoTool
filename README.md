# CBIR Web Annotation Tools for XCloud

<p align="left"><img src="logo/horizontal.svg" alt="XCloud" height="120px"></p>

## Introduction
An online annotation toolkit with web-based UI for ```Image Retrieval/ReID/Face Recognition``` tasks. It is freely accessible to **both research and industrial** fields.

**Note**: training and testing codes can be found from [XCloud](https://github.com/lucasxlu/XCloud/tree/master/research/cbir). 

![index](./index.png)

### Backbone
| Architecture | Supervision | Status |
| :---: |:---: |:---: |
| DenseNet121 | Softmax | [YES] |
| DenseNet121 | CenterLoss | [YES] |
| DenseNet121 | A-Softmax | [YES] |
| ResNeXt50 | A-Softmax | [TODO] |
| SEResNeXt50 | A-Softmax | [TODO] |


### Dependency
 * [Faiss](https://github.com/facebookresearch/faiss.git)
 * [Django](https://www.djangoproject.com/)


## How to use
1. Train your embedding model with the code provided in [XCloud](https://github.com/lucasxlu/XCloud.git) [cbir branch](https://github.com/lucasxlu/XCloud/tree/master/research/cbir).
2. Extract deep features with the code provided in [preprocess](cv/preprocess), [CbirAnnoTool](https://github.com/lucasxlu/CbirAnnoTool.git) support [SoftmaxLoss](cv/preprocess/ext_feats_softmaxloss.py),
[CenterLoss](cv/preprocess/ext_feats_centerloss.py) and [ASoftmaxLoss](cv/preprocess/ext_feats_asoftmaxloss.py). 
3. Start Django service by ```python3 manage.py runserver 0.0.0.0:8001```
4. Open your browser and visit **http://YOUR_MACHINE_IP:8001/cv/annoview**


## Citation
This tool is supplementary of [XCloud](https://github.com/lucasxlu/XCloud.git), If you use this tool in your research, please cite our [technical report](https://lucasxlu.github.io/blog/about/XCloud.pdf) about [XCloud](https://github.com/lucasxlu/XCloud.git) as:
```
@article{xu2019xcloud,
  title={XCloud: Design and Implementation of AI Cloud Platform with RESTful API Service},
  author={Xu, Lu and Wang, Yating},
  journal={arXiv preprint arXiv:1912.10344},
  year={2019}
}
```


## License
[MIT](./LICENSE)