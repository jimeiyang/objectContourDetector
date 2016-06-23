#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
void TensorProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  K_ = bottom[0]->count() / bottom[0]->num();
  L_ = bottom[1]->count() / bottom[1]->num();

  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The num of two bottom layers does not match.";
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, L_, N_, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, L_, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  S_.Reshape(1, L_, N_, K_);
  LN_.Reshape(1, 1, L_, N_);
  NK_.Reshape(1, 1, N_, K_);
}

template <typename Dtype>
void TensorProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The num of two bottom layers does not match.";
  (*top)[0]->Reshape(bottom[0]->num(), N_, 1, 1);
  
  T_.Reshape(bottom[0]->num(), 1, L_, N_);
  // Set up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);
    caffe_set(M_, (Dtype)1., bias_multiplier_.mutable_cpu_data());
    bias_multiplier2_.Reshape(1, 1, M_, N_);
    caffe_set(M_*N_, (Dtype)1., bias_multiplier2_.mutable_cpu_data());
  }
}

template <typename Dtype>
void TensorProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* T = T_.mutable_cpu_data();

  CHECK_EQ(L_, bottom[1]->count() / bottom[1]->num()) 
    << "Second bottom channel does not match";
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, L_*N_, K_, 
      (Dtype)1., bottom_data, weight, (Dtype)0., T);
  for (int i = 0; i < M_; ++i) {
    caffe_cpu_gemv<Dtype>(CblasTrans, L_, N_, (Dtype)1.,
        T + i*L_*N_, bottom_data2 + i*L_, (Dtype)0., top_data + i*N_);
  }
  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, L_, (Dtype)1.,
        bottom_data2, bias, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void TensorProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* bottom_data2 =  (*bottom)[1]->cpu_data();
    Dtype* w_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* s_diff = S_.mutable_cpu_diff();

    // Gradient with respect to weight
    caffe_set<Dtype>(L_*N_*K_, (Dtype)0., w_diff);
    for (int i = 0; i < M_; ++i) {
      caffe_set<Dtype>(L_*N_*K_, (Dtype)0., s_diff);
      for (int l = 0; l < L_; ++l) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, 
            bottom_data2[i*L_ + l], top_diff + i*N_, bottom_data + i*K_,
            (Dtype)1., s_diff + l*N_*K_);
      }
      caffe_add<Dtype>(L_*N_*K_, s_diff, w_diff, w_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data2 = (*bottom)[1]->cpu_data();
    // Gradient with respect to bias
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L_, N_, M_, (Dtype)1.,
        bottom_data2, top_diff, (Dtype)0., this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* bottom_data2 = (*bottom)[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* LN = LN_.mutable_cpu_diff();

    // Gradient with respect to bottom data
    for (int i = 0; i < M_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L_, N_, 1, (Dtype)1.,
          bottom_data2 + i*L_, top_diff + i*N_, (Dtype)0., LN);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, 1, N_*L_, (Dtype)1.,
          this->blobs_[0]->cpu_data(), LN, (Dtype)0., bottom_diff + i*K_);
    }
  }
  if (propagate_down[1]) { 
    Dtype* bottom_diff2 = (*bottom)[1]->mutable_cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bias = this->blobs_[1]->cpu_data();
    Dtype* NK = NK_.mutable_cpu_diff();

    // Gradient with respect to bottom data
    for (int i = 0; i < M_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, (Dtype)1.,
          top_diff + i*N_, bottom_data + i*K_, (Dtype)0., NK);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, L_, 1, N_*K_, (Dtype)1.,
          this->blobs_[0]->cpu_data(), NK, (Dtype)0., bottom_diff2 + i*L_);
    }

    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, L_, N_, (Dtype)1.,
          top_diff, bias, (Dtype)1., bottom_diff2);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TensorProductLayer);
#endif

INSTANTIATE_CLASS(TensorProductLayer);

}  // namespace caffe
