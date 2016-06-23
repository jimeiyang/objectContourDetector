#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";

  kernel_size_ = this->layer_param_.local_weighted_convolution_param().kernel_size();
  stride_ = this->layer_param_.local_weighted_convolution_param().stride();
  pad_ = this->layer_param_.local_weighted_convolution_param().pad();
  num_ = bottom[0]->num();
  channels_input_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_output_ = this->layer_param_.local_weighted_convolution_param().num_output();
  CHECK_GT(channels_output_, 0); 
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  height_out_ = stride_ * (height_ - 1) + kernel_size_ - 2 * pad_; 
  width_out_ = stride_ * (width_ - 1) + kernel_size_ - 2 * pad_;
  col_buffer_.Reshape(1, channels_output_ * kernel_size_ * kernel_size_, height_, width_);
  // Set the parameters
  bias_term_ = this->layer_param_.local_weighted_convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  K_ = channels_output_ * kernel_size_ * kernel_size_;
  N_ = height_ * width_;
  M_ = channels_input_;
  (*top)[0]->Reshape(bottom[0]->num(), channels_output_, height_out_, width_out_);
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
    this->blobs_[0].reset(new Blob<Dtype>(1, K_, M_, N_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.local_weighted_convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_output_, height_out_ * width_out_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.local_weighted_convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());  
    }
  }
}

template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top) {

  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  /*
  Blob<Dtype> E;
  E.Reshape(1, 1, 1, M_);
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(&E);
  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, M_, N_);
  */
  
  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, 1, N_);
  for (int n=0; n<num_; n++) {
    
    // compute patches by inner product
    for (int k=0; k<K_; k++) {
      for (int m=0; m<M_; m++) {
        caffe_mul(N_, bottom_data + bottom[0]->offset(n,m), weight + this->blobs_[0]->offset(0,k,m), intermediate.mutable_cpu_data());
        caffe_cpu_axpby(N_, Dtype(1.0), intermediate.cpu_data(), Dtype(1.0), col_data + col_buffer_.offset(0,k));
      }
      /*
      caffe_mul(M_*N_, bottom_data + bottom[0]->offset(n), weight + this->blobs_[0]->offset(0,k), intermediate.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                            (Dtype)1., E.cpu_data(),
                            intermediate.cpu_data(),
                            (Dtype)0., col_data + col_buffer_.offset(0,k));
      */
    }
    // col2im back to the data
    col2im_cpu(col_data, channels_output_, height_out_, width_out_, kernel_size_, kernel_size_,
                 pad_, pad_, stride_, stride_, top_data + (*top)[0]->offset(n));
   
    // add up bias term
    if (bias_term_) {
      caffe_add(channels_output_ * height_out_ * width_out_, this->blobs_[1]->cpu_data(),
                top_data + (*top)[0]->offset(n), top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
void LocalWeightedDeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = NULL;

  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, 1, N_);

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n=0; n<num_; n++) {
      caffe_add(channels_input_ * N_, bias_diff, top_diff + top[0]->offset(n), bias_diff);
    }
  }

  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  for (int n=0; n<num_; n++) {

    // crop patches from top_diff
    im2col_cpu(top_diff + top[0]->offset(n), channels_output_, height_out_, width_out_,
               kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, col_diff);

    // gradient wrt weight by outer product
    for (int k=0; k<K_; k++) {
      for (int m=0; m<M_; m++) {
        caffe_mul(N_, col_diff + col_buffer_.offset(0,k),  
                  bottom_data + (*bottom)[0]->offset(n,m), weight_diff + this->blobs_[0]->offset(0,k,m));
      }
    }
      
    // gradient wrt bottom data by inner product
    if (propagate_down[0]) {
      for (int k=0; k<K_; k++) {
        for (int m=0; m<M_; m++) {
          caffe_mul(N_, col_diff + col_buffer_.offset(0,k), weight + this->blobs_[0]->offset(0,k,m), intermediate.mutable_cpu_data());
          caffe_cpu_axpby(N_, Dtype(1.0), intermediate.cpu_data(), Dtype(1.0), bottom_diff + (*bottom)[0]->offset(n,m));
        }
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(LocalWeightedDeconvolutionLayer);
#endif

INSTANTIATE_CLASS(LocalWeightedDeconvolutionLayer);

}  // namespace caffe
