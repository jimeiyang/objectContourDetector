#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/stochastic_neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BernoulliSample(const int n, const Dtype* probs, Dtype* samples) {
  CUDA_KERNEL_LOOP(index, n) {
    samples[index] = (probs[index] > samples[index]) ? 1 : 0; 
  }
}

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Sample from uniform distribution
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1), top_data);  
  // Generate bernoulli samples
  const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
  BernoulliSample<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sigmoid_output_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SigmoidBernoulliBackward(const int n, const Dtype* top_diff, 
    const Dtype* sigmoid_output, const Dtype* samples, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype diff = sigmoid_output[index] - samples[index];
    bottom_diff[index] = top_diff[index] * diff;
  }
}

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    SigmoidBernoulliBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    	count, top_diff, sigmoid_output_data, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK; 
  }
}

INSTANTIATE_CLASS(SigmoidBernoulliLayer);


}  // namespace caffe
