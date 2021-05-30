# Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection
This is the official implementation of:

Update(May 30): Please use the lastest parallelized version.

**Zeyi Huang', Yang Zou', Vijayakumar Bhagavatula, and Dong Huang**, ***Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection***, **NeurIPS 2020**, [Arxiv version](https://arxiv.org/pdf/2010.12023.pdf).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/comprehensive-attention-self-distillation-for/weakly-supervised-object-detection-on-mscoco)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-mscoco?p=comprehensive-attention-self-distillation-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/comprehensive-attention-self-distillation-for/weakly-supervised-object-detection-on-pascal)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-pascal?p=comprehensive-attention-self-distillation-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/comprehensive-attention-self-distillation-for/weakly-supervised-object-detection-on-pascal-1)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-pascal-1?p=comprehensive-attention-self-distillation-for)

### Citation: 

```bash
@article{huang2020comprehensive,
  title={Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection},
  author={Huang, Zeyi and Zou, Yang and Kumar, BVK and Huang, Dong},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Installation

### Requirements

- Python == 3.7
- Pytorch == 1.1.0
- Torchvision == 0.3.0
- Cuda == 10.0
- cython
- scipy
- sklearn
- opencv
- GPU: TITAN RTX (24G of memory)

Note: To train with GPU of small memory, CASD_IW is partially parallelized. Fully parallelized version is coming soon. Thanks for your patience.

### Preparation

1. Clone the repository
```bash
git clone https://github.com/DeLightCMU/CASD.git
```

2. Compile the CUDA code
```bash
cd CASD/lib
bash make_cuda.sh
```

3. Download the training, validation, test data and the VOCdevkit
```bash
mkdir data
cd data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
```

4. Extract all of these tars into one directory named VOCdevkit
```bash
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

5. Create symlinks for PASCAL VOC dataset
```bash
cd CASD/data
ln -s VOCdevkit VOCdevkit2007
```

6. Download pretrained ImageNet weights from [here](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), and put it in the data/imagenet_weights/

7. Download selective search proposals from [here](https://drive.google.com/drive/folders/1dAH1oPZHKGWowOFVewblSQDJzKobTR5A), and put it in the data/selective_search_data/

### Training and Testing

Train a vgg16 Network on VOC 2007 trainval
```bash
bash experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
```

Test a vgg16 Network on VOC 2007 test
```bash
bash experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
```
Download log and weight from [here](https://drive.google.com/drive/folders/1p7iCBzp1HvAeLuW9RgTGB9X_meC58S_8?usp=sharing)

## Acknowledgement
We borrowed code from [MLEM](https://github.com/vasgaowei/pytorch_MELM), [PCL](https://github.com/ppengtang/pcl.pytorch), and [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch).
