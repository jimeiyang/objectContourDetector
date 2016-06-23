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
void DownsamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  DownsamplingParameter downsample_param = this->layer_param_.downsampling_param();
  CHECK(!downsample_param.has_kernel_size() !=
      !(downsample_param.has_kernel_h() && downsample_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(downsample_param.has_kernel_size() ||
      (downsample_param.has_kernel_h() && downsample_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  if (downsample_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = downsample_param.kernel_size();
  } else {
    kernel_h_ = downsample_param.kernel_h();
    kernel_w_ = downsample_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
}

template <typename Dtype>
void DownsamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  downsampled_height_ = static_cast<int>(ceil(static_cast<float>(height_) / kernel_h_));
  downsampled_width_ = static_cast<int>(ceil(static_cast<float>(width_) / kernel_w_));
  (*top)[0]->Reshape(bottom[0]->num(), channels_, downsampled_height_,
      downsampled_width_);
}

template <typename Dtype>
void DownsamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int dh = 0; dh < downsampled_height_; ++dh) {
          for (int dw = 0; dw < downsampled_width_; ++dw) {
            int hstart = dh * kernel_h_;
            int wstart = dw * kernel_w_;
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            top_data[dh * downsampled_width_ + dw] +=
                    bottom_data[hstart * width_ + wstart];
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
}

template <typename Dtype>
void DownsamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
        for (int dh = 0; dh < downsampled_height_; ++dh) {
          for (int dw = 0; dw < downsampled_width_; ++dw) {
	    int h = static_cast<int>(static_cast<float>(dh) * kernel_h_);
	    int w = static_cast<int>(static_cast<float>(dw) * kernel_w_);
            bottom_diff[h * width_ + w] = top_diff[dh * downsampled_width_ + dw];
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
}


#ifdef CPU_ONLY
STUB_GPU(DownsamplingLayer);
#endif

INSTANTIATE_CLASS(DownsamplingLayer);


}  // namespace caffe
