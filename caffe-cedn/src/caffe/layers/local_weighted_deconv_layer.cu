#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/local_update.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/// @brief refer to GPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  for (int n=0; n<num_; n++) {
    
    // clear memory for col_data
    CUDA_CHECK(cudaMemset(col_data, 0, sizeof(Dtype) * col_buffer_.count())); 

    // compute patches by inner product
    local_update3_gpu(bottom_data + bottom[0]->offset(n), weight, 
                      col_data, M_, N_, K_);

    // col2im back to the data
    col2im_gpu(col_data, channels_output_, height_out_, width_out_, kernel_size_, kernel_size_,
                 pad_, pad_, stride_, stride_, top_data + (*top)[0]->offset(n));
   
    // add up bias term
    if (bias_term_) {
      caffe_gpu_add(channels_output_ * height_out_ * width_out_, this->blobs_[1]->gpu_data(),
                top_data + (*top)[0]->offset(n), top_data + (*top)[0]->offset(n));
    }
  }

}

/// @brief refer to GPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n=0; n<num_; n++) {
      caffe_gpu_add(channels_input_ * N_, bias_diff, top_diff + top[0]->offset(n), bias_diff);
    }
  }

  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count()));
  CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype) * col_buffer_.count()));
  for (int n=0; n<num_; n++) {

    // crop patches from top_diff
    im2col_gpu(top_diff + top[0]->offset(n), channels_output_, height_out_, width_out_,
               kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, col_diff);

    // gradient wrt weight by outer product
    local_update1_gpu(col_diff, bottom_data + (*bottom)[0]->offset(n), weight_diff, 
		      M_, N_, K_);
    // gradient wrt bottom data by inner product
    if (propagate_down[0]) {
      local_update2_gpu(col_diff, weight, bottom_diff + (*bottom)[0]->offset(n),
                        M_, N_, K_);
    }

  }

}


INSTANTIATE_CLASS(LocalWeightedDeconvolutionLayer);

}  // namespace caffe
