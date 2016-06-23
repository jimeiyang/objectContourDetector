# objectContourDetector

This is the code for arXiv paper [Object Contour Detection with a Fully Convolutional Encoder-Decoder Network](http://arxiv.org/abs/1603.04530) by Jimei Yang, Brian Price, Scott Cohen, Honglak Lee and Ming-Hsuan Yang, 2016.


## Contents
* This code includes 
 - the Caffe toolbox for Convolutional Encoder-Decoder Networks (`caffe-cedn`)
 - scripts for training and testing the PASCAL object contour detector, and 
 - scripts to refine segmentation anntations based on dense CRF.
* It is tested on Linux (Ubuntu 14.04) with NVIDIA TITAN X GPU.

Please follow the instructions below to run the code.

## Compilation
* Compile the `Caffe`, `matcaffe` and `pycaffe` in the `caffe-cedn` package. 

## Training on PASCAL
* Download the pre-processed dataset by running the script
```
./data/PASCAL/get_pascal_training_data.sh
```
* Download the VGG16 net for initialization by running the script
```
./models/get_vgg16_net.sh
```
* Start training by running the script
```
./code/train.sh
```
* Test the learned network by running the script
```
./code/test.sh
```

## Testing the pre-trained model
* Download the pre-trained model by running the script
```
./models/PASCAL/get_pretrained_pascal_net.sh
```

## Citation

If you find this useful, please cite our work as follows:
```
@inproceedings{yang2016object,
  title={Object Contour Detection with a Fully Convolutional Encoder-Decoder Network},
  author={Yang, Jimei and Price, Brian and Cohen, Scott and Lee, Honglak and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:1603.04530},
  year={2016}
}
```

Please contact "jimyang@adobe.com" if any questions. 
