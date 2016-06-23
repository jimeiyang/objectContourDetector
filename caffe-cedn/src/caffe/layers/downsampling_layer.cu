#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DownsampleForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int downsampled_height, const int downsampled_width,
    const int kernel_h, const int kernel_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int dw = index % downsampled_width;
    int dh = (index / downsampled_width) % downsampled_height;
    int c = (index / downsampled_width / downsampled_height) % channels;
    int n = index / downsampled_width / downsampled_height / channels;
    bottom_data += (n * channels + c) * height * width;
    int w = static_cast<int>(static_cast<float>(dw) * kernel_w); 
    int h = static_cast<int>(static_cast<float>(dh) * kernel_h);
    w = min(w, width);
    h = min(h, height);
    top_data[index] = bottom_data[h * width + w];
  }
}


template <typename Dtype>
void DownsamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
  DownsampleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, downsampled_height_, downsampled_width_, kernel_h_, kernel_w_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void DownsampleBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int downsampled_height, const int downsampled_width,
    const int kernel_h, const int kernel_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    top_diff += (n * channels + c) * downsampled_height * downsampled_width;
    if (w % kernel_w == 0 && h % kernel_h == 0) {
	int dw = static_cast<int>(static_cast<float>(w) / kernel_w); 
	int dh = static_cast<int>(static_cast<float>(h) / kernel_h);
    	bottom_diff[index] = top_diff[dh * downsampled_width + dw];
    }
    else {
	bottom_diff[index] = 0;
    }
  }
}


template <typename Dtype>
void DownsamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  DownsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), channels_,
      height_, width_, downsampled_height_, downsampled_width_, kernel_h_, kernel_w_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(DownsamplingLayer);


}  // namespace caffe
