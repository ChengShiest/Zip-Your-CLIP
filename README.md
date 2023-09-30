# Zip-Your-CLIP

This repository is the official implementation of [Zip](https://arxiv.org/abs/2309.14494).

**[The Devil is in the Object Boundary: Towards Annotation-free Instance Segmentation using Foundation Models](https://arxiv.org/abs/2309.14494)**

[Cheng Shi](https://chengshiest.github.io/), [Sibei Yang†](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

†Corresponding Author

<!-- [![arXiv](https://img.shields.io/badge/arXiv-FreeBloom-b31b1b.svg)](https://arxiv.org/abs/2309.14494) ![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch) -->

<!-- Code will be released soon, stay tuned! -->

![image-20230924124604776](__assets__/fig1.png)


## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```


Install CLIP and CLIP_Surgery
```
pip install git+https://github.com/openai/CLIP.git

git clone https://github.com/xmed-lab/CLIP_Surgery.git
```

Make sure that you correctly install the dependencies of CLIP_Surgery
```
### Test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("CS-ViT-B/16", device=device)
```

## <a name="GettingStarted"></a>Getting Started
![image-20230924124604777](__assets__/fig2.png)


### Clustering Results


## License

This Project is licensed under the [Apache 2.0 license](__assets__/LICENSE.txt).


## Citation

```
@article{zip,
	title={The devil is in the object boundary: towards annotation-free instance segmentation using Foundation Modelsr},
	author={Shi, Cheng and Yang, Sibei},
	journal={arXiv preprint arXiv:2309.14494},
	year={2023}
}
```