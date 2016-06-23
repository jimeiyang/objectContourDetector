#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/stochastic_neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GaussianForward(const int n, const Dtype* mu, Dtype* noise, Dtype* sample) {
  CUDA_KERNEL_LOOP(index, n) {
    sample[index] = mu[index] + noise[index];
  }
}

template <typename Dtype>
__global__ void GaussianForward(const int n, const Dtype* mu, const Dtype* log_var, Dtype* noise, Dtype* sample) {
  CUDA_KERNEL_LOOP(index, n) {
    sample[index] = mu[index] + exp(.5*log_var[index])*noise[index];
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* mu = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* noise_data = noise_.mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_gpu_rng_gaussian(count, (Dtype)0.0, sigma_, noise_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (bottom.size() > 1) {
    const Dtype* log_var = bottom[1]->gpu_data();
    GaussianForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, mu, log_var, noise_data, top_data);
  }
  else {
    GaussianForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, mu, noise_data, top_data);
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GaussianBackwardMu(const int n, const Dtype* in_diff, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index];
  }
}

template <typename Dtype>
__global__ void GaussianBackwardLogVar(const int n, const Dtype* top_diff, const Dtype* noise, const Dtype* log_var, Dtype* log_var_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    log_var_diff[index] = .5*top_diff[index]*noise[index]*exp(.5*log_var[index]);
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* mu_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    GaussianBackwardMu<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mu_diff);
    if (bottom->size() > 1) { 
      const Dtype* noise_data = noise_.gpu_data();
      const Dtype* log_var = (*bottom)[1]->gpu_data();
      Dtype* log_var_diff = (*bottom)[1]->mutable_gpu_diff();
      GaussianBackwardLogVar<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, noise_data, log_var, log_var_diff);
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(GaussianLayer);


}  // namespace caffe
