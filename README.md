# (PHFNet)Fast and Effective: Progressive Hierarchical Fusion Classification for Remote Sensing Images

> Xueli Geng, Licheng Jiao, Lingling Li, Xu Liu, Fang Liu, Shuyuan Yang
> *IEEE Transactions on Multimedia, 2024*


## Abstract

Multisource remote sensing image fusion classification aims to produce accurate pixel-level classification maps by
combining complementary information from different sources of
remote sensing data. Existing methods based on Convolutional
Neural Networks (CNN-based) utilize a patch-based learning
framework, which has a high computational cost, leading to
poor real-time performance. In contrast, methods based on
Fully Convolutional Networks (FCN-based) can process the
entire image directly, achieving fast inference. However, FCNbased methods require high computational resources and exhibit
shortcomings in feature fusion, hindering practical applications.
In this paper, a lightweight FCN-based Progressive Hierarchical
Fusion Network (PHFNet) is tailored for multisource remote
sensing image classification. PHFNet comprises a pyramid dualpath encoder and a pyramid decoder. In the encoder, cross-source
features are hierarchically fused via the adaptive modulation
fusion module (AMF), which leverages style calibration for
cross-source alignment and promotes the complementarity of
the fusion feature. In the decoder, we introduced an improved
convolutional gated recurrent unit (iConvGRU) to progressively
integrate the semantic and detailed information of hierarchical
features, producing a context-enhanced global representation. In
addition, we consider the relation between the channel number,
convolutional kernel size, and parameter count to make the model
as lightweight as possible. Comprehensive evaluations on three
multisource remote sensing datasets demonstrate that PHFNet
improves overall accuracy by 1.5% to 2.8% with a low computational overhead compared to state-of-the-art methods.
[[paper]](https://ieeexplore.ieee.org/document/10522846). 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites

In order to run this implementation, you need to have the following software and libraries installed:

- Python 3.7
- PyTorch 1.3
- CUDA (if using GPU)
- NumPy
- Matplotlib
- OpenCV


### Training the Model

To train the model, you can run the following command:

```
python main.py
```

If you have any questions, please contact us (xlgeng@stu.xidian.edu.cn)



## Citation

Please cite our paper if you find this code useful for your research.

```
@ARTICLE{10522846,
  author={Geng, Xueli and Jiao, Licheng and Li, Lingling and Liu, Xu and Liu, Fang and Yang, Shuyuan},
  journal={IEEE Transactions on Multimedia}, 
  title={Fast and Effective: Progressive Hierarchical Fusion Classification for Remote Sensing Images}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMM.2024.3398371}}
```
