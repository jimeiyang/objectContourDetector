#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/stochastic_neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Sample from sigmoid activations
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  caffe_rng_uniform(count, Dtype(0), Dtype(1), top_data);
  for (int i = 0; i < count; ++i) {
    top_data[i] = (sigmoid_output_data[i] > top_data[i]) ? 1 : 0;
  }
}

template <typename Dtype>
void SigmoidBernoulliLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* samples = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, samples, bottom_diff);
    caffe_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidBernoulliLayer);
#endif

INSTANTIATE_CLASS(SigmoidBernoulliLayer);


}  // namespace caffe
