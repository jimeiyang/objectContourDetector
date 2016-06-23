#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TensorProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data2 = bottom[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* T = T_.mutable_gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, L_*N_, K_, 
      (Dtype)1., bottom_data, weight, (Dtype)0., T);
  for (int i = 0; i < M_; ++i) {
    caffe_gpu_gemv<Dtype>(CblasTrans, L_, N_, (Dtype)1.,
        T + i*L_*N_, bottom_data2 + i*L_, (Dtype)0., top_data + i*N_);
  }
  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, L_, (Dtype)1.,
        bottom_data2, bias, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void TensorProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* bottom_data2 =  (*bottom)[1]->gpu_data();
    Dtype* w_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* s_diff = S_.mutable_gpu_diff();

    // Gradient with respect to weight
    caffe_gpu_set<Dtype>(L_*N_*K_, (Dtype)0., w_diff);
    for (int i = 0; i < M_; ++i) {
      caffe_gpu_set<Dtype>(L_*N_*K_, (Dtype)0., s_diff);
      for (int l = 0; l < L_; ++l) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, 
            (Dtype)1., top_diff + i*N_, bottom_data + i*K_,
            (Dtype)1., s_diff + l*N_*K_);
      }
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, L_, N_*K_, 1, 
          (Dtype)1., bottom_data2 + i*L_, s_diff, 
          (Dtype)0., s_diff);
      caffe_gpu_add<Dtype>(L_*N_*K_, s_diff, w_diff, w_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data2 = (*bottom)[1]->gpu_data();
    // Gradient with respect to bias
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L_, N_, M_, (Dtype)1.,
        bottom_data2, top_diff, (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const Dtype* bottom_data2 = (*bottom)[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* LN = LN_.mutable_gpu_diff();

    // Gradient with respect to bottom data
    for (int i = 0; i < M_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L_, N_, 1, (Dtype)1.,
          bottom_data2 + i*L_, top_diff + i*N_, (Dtype)0., LN);
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, 1, N_*L_, (Dtype)1.,
          this->blobs_[0]->gpu_data(), LN, (Dtype)0., bottom_diff + i*K_);
    }
  }
  if (propagate_down[1]) { 
    Dtype* bottom_diff2 = (*bottom)[1]->mutable_gpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bias = this->blobs_[1]->gpu_data();
    Dtype* NK = NK_.mutable_gpu_diff();

    // Gradient with respect to bottom data
    for (int i = 0; i < M_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, 1, (Dtype)1.,
          top_diff + i*N_, bottom_data + i*K_, (Dtype)0., NK);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, L_, 1, N_*K_, (Dtype)1.,
          this->blobs_[0]->gpu_data(), NK, (Dtype)0., bottom_diff2 + i*L_);
    }

    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, L_, N_, (Dtype)1.,
          top_diff, bias, (Dtype)1., bottom_diff2);
    }
  }
}

INSTANTIATE_CLASS(TensorProductLayer);

}  // namespace caffe
