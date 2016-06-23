#!/usr/bin/env python

# --------------------------------------------------------
# Convolutional Encoder-Decoder Networks for Contour Detection 
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
See README.md for installation instructions before running.
"""

import _init_paths
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.blob import im_list_to_blob

# PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

TEST_MAX_SIZE = 512

ROOT_DIR = '../'

NETS = {'pascal': ('PASCAL', 'vgg-16-encoder-decoder-contour-w10-pascal', 'iter030')}

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image
    """
    im_orig = im;
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    # Prevent the biggest axis from being more than MAX_SIZE
    if im_size_max > TEST_MAX_SIZE:
        im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

    # Pad the image to the standard size (optional)
    top = (TEST_MAX_SIZE - im_shape[0]) / 2 
    bottom = TEST_MAX_SIZE - im_shape[0] - top
    left = (TEST_MAX_SIZE - im_shape[1]) / 2 
    right = TEST_MAX_SIZE - im_shape[1] - left
    im_pad = (top, bottom, left, right)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    # substract the color mean
    im = im.astype(np.float32, copy=True)
    im -= PIXEL_MEANS

    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_shape, im_pad

def _get_contour_map(probmap, im_shape, im_pad):
    """Convert network output to contour maps."""
     
    probmap = np.squeeze(probmap) 

    top = im_pad[0]
    bottom = im_pad[1]
    left = im_pad[2]
    right = im_pad[3]
    probmap = probmap[top:-bottom, left:-right]

    height = im_shape[0]
    width = im_shape[1]
    probmap = cv2.resize(probmap, (im_shape[1], im_shape[0])) 

    return probmap

def contour_detection(net, im):
    """Detect object contours."""

    # Detect object contours
    timer = Timer()
    timer.tic()

    # Convert image to network blobs
    blobs = {'data': None}
    blobs['data'], im_shape, im_pad = _get_image_blob(im) 
    # Reshape network inputs
#    net.blobs['data'].reshape(*(blobs['data'].shape))
    # Run forward inference
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))
    probmap = blobs_out['probmap']    
    # Convert network output to image
    probmap = _get_contour_map(probmap, im_shape, im_pad) 

    timer.toc()
    print ('Detection took {:.3f}s').format(timer.total_time)
    
    return probmap    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Run a Convolutional Encoder-Decoder network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='infer_net', help='Network to use [pascal]',
                        choices=NETS.keys(), default='pascal')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(ROOT_DIR, 'models', 'PASCAL',
                            '{:s}-test.prototxt'.format(NETS[args.infer_net][1]))
    caffemodel = os.path.join(ROOT_DIR, 'models', 'PASCAL',
                              '{:s}-{:s}.caffemodel'.format(NETS[args.infer_net][1], NETS[args.infer_net][2]))

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    net = caffe.Net(prototxt, caffemodel)
    net.set_phase_test()
    print 'Net is initialized.\n'

    if args.cpu_mode:
        net.set_mode_cpu()
    else:
        net.set_mode_gpu()
        net.set_device(args.gpu_id)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    list_file = os.path.join(ROOT_DIR, '../data/PASCAL', 'val.txt')
    f = open(list_file, 'r')
    imnames = f.readlines()    
    for name in imnames:
    	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    	print 'Infer for {}'.format(name)
        # Load the test image
        name = name[:-1]
        im_file = os.path.join(ROOT_DIR, '../data/PASCAL', 'JPEGImages', name + '.jpg')
        im = cv2.imread(im_file)
	# Run CEDN inference
    	probmap = contour_detection(net, im)
        # Save detections
        res_file = os.path.join(ROOT_DIR, '../results/PASCAL', name + '.png')
	cv2.imwrite(res_file, (255*probmap).astype(np.uint8, copy=True))

