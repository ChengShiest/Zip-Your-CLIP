<br />
<p align="center">
  <h1 align="center">Zip-Your-CLIP</h1>
  <p align="center">
	Zip: CLIP Itself is a Good Object-detector
    ICLR, 2024
    <br />
    <a href="https://chengshiest.github.io/"><strong>Cheng Shi</strong></a>
    ·
    <a href="https://faculty.sist.shanghaitech.edu.cn/yangsibei/"><strong>Sibei Yang†</strong></a>
  </p>

  <p align="center">
    <a href='https://openreview.net/forum?id=4JbrdrHxYy'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
  </p>
<br />

<!-- **[The Devil is in the Object Boundary: Towards Annotation-free Instance Segmentation using Foundation Models](https://openreview.net/forum?id=4JbrdrHxYy&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))**

[Cheng Shi](https://chengshiest.github.io/), [Sibei Yang†](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

†Corresponding Author -->

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
@inproceedings{
shi2024the,
title={The Devil is in the Object Boundary: Towards Annotation-free Instance Segmentation using Foundation Models},
author={Cheng Shi and Sibei Yang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=4JbrdrHxYy}
}
```
