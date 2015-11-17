# CNN-Prediction
A fast version of cnn prediction
Convolutional Neural Network (CNN) is a great tool to solve computer vision and image processing problem. 
Many times, it works like a charm. However, under certain scenario, dense scanning, applying cnn model on 
each image pixel, is really computational expensive.

This project implemented a fast version of dense scanning, it has the same result as exactly sliding window (stride is 1),
and much faster. It has following features:

1. The model is trained from image patches. For example 51 x 51 image patch in the example

2. The filters learned are applied on whole image

3. Specially handling is taken care in the forward part of max-pooling layer to avoid misalignment

4. The result is the same as stride 1 based sliding window




For more details, check the original paper:
@article{giusti2013fast,
  title={Fast image scanning with deep max-pooling convolutional neural networks},
  author={Giusti, Alessandro and Cire{\c{s}}an, Dan C and Masci, Jonathan and Gambardella, Luca M and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1302.1700},
  year={2013}
}

