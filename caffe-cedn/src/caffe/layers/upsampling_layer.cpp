#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void UpsamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  UpsamplingParameter upsample_param = this->layer_param_.upsampling_param();
  CHECK(!upsample_param.has_kernel_size() !=
      !(upsample_param.has_kernel_h() && upsample_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(upsample_param.has_kernel_size() ||
      (upsample_param.has_kernel_h() && upsample_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  if (upsample_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = upsample_param.kernel_size();
  } else {
    kernel_h_ = upsample_param.kernel_h();
    kernel_w_ = upsample_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  upsampled_height_ = static_cast<int>(static_cast<float>(height_) * kernel_h_);
  upsampled_width_ = static_cast<int>(static_cast<float>(width_) * kernel_w_);
  (*top)[0]->Reshape(bottom[0]->num(), channels_, upsampled_height_,
      upsampled_width_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int uh = 0; uh < upsampled_height_; ++uh) {
          for (int uw = 0; uw < upsampled_width_; ++uw) {
	    if (uh % kernel_h_ == 0 && uw % kernel_w_ == 0) { 
	      int sh = static_cast<int>(static_cast<float>(uh) / kernel_h_); 
	      int sw = static_cast<int>(static_cast<float>(uw) / kernel_w_);
              top_data[uh * upsampled_width_ + uw] = bottom_data[sh * width_ + sw];
	    }
	    else {
	      top_data[uh * upsampled_width_ + uw] = 0;
	    }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int sh = 0; sh < height_; ++sh) {
          for (int sw = 0; sw < width_; ++sw) {
	    int uh = static_cast<int>(static_cast<float>(sh) * kernel_h_);
	    int uw = static_cast<int>(static_cast<float>(sw) * kernel_w_);
            bottom_diff[sh * width_ + sw] = top_diff[uh * upsampled_width_ + uw];
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
}


#ifdef CPU_ONLY
STUB_GPU(UpsamplingLayer);
#endif

INSTANTIATE_CLASS(UpsamplingLayer);


}  // namespace caffe
