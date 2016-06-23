# objectContourDetector

This is the code for arXiv paper [Attribute2Image: Conditional Image Generation from Visual Attributes](https://arxiv.org/abs/1512.00570) by Xinchen Yan, Jimei Yang, Kihyuk Sohn and Honglak Lee.

Please follow the instructions to run the code.

## Requirements
Attribute2Image requires or works with
* Mac OS X or Linux
* NVIDIA GPU

## Installing Dependency
* Install [Torch](http://torch.ch)

## Data Preprocessing
* For LFW dataset, please run the script to download the pre-processed dataset
```
./prepare_cropped_lfw.sh
```

* For CelebA dataset, please download the [original dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and then run the script for pre-processing 
```
./prepare_cropped_celeba.sh
```
## Training
* If you want to train the LFW image generator, please run the script (less than 3 hours)
```
./demo_lfw.sh
```
* If you want to train the CelebA image generator, please run the script (around 24 hours)
```
./demo_celeba.sh
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
