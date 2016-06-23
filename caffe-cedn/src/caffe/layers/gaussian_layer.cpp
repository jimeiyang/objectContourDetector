#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/stochastic_neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GaussianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  sigma_ = this->layer_param_.gaussian_param().sigma();
  CHECK_GE(sigma_, 0) << "sigma should be >= 0";
}

template <typename Dtype>
void GaussianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num, bottom[i]->num());
    CHECK_EQ(channels, bottom[i]->channels());
    CHECK_EQ(height, bottom[i]->height());
    CHECK_EQ(width, bottom[i]->width());
  }
  (*top)[0]->Reshape(num, channels, height, width);
  // Set up the cache for random number generation
  noise_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void GaussianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* mu = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* noise_data = noise_.mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_rng_gaussian(count, (Dtype)0, sigma_, noise_data); 
  if (bottom.size() > 1) {
    const Dtype* log_var = bottom[1]->cpu_data();
    for (int i = 0; i < count; ++i) {
      top_data[i] = mu[i] + exp(.5*log_var[i])*noise_data[i];
    }
  }
  else {
    for (int i = 0; i < count; ++i) {
      top_data[i] = mu[i] + noise_data[i];
    }
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* mu_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      mu_diff[i] = top_diff[i];
    }
    if (bottom->size() > 1) {
      const Dtype* noise_data = noise_.cpu_data();
      const Dtype* log_var = (*bottom)[1]->cpu_data();
      Dtype* log_var_diff = (*bottom)[1]->mutable_cpu_diff();
      for (int i = 0; i < count; ++i) {
        log_var_diff[i] = .5*top_diff[i]*noise_data[i]*exp(.5*log_var[i]);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GaussianLayer);
#endif

INSTANTIATE_CLASS(GaussianLayer);


}  // namespace caffe
