#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FoldingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  FoldingParameter folding_param = this->layer_param_.folding_param();
  channels_folded_ = folding_param.channels_folded();
  height_folded_ = folding_param.height_folded();
  width_folded_ = folding_param.width_folded();
}

template <typename Dtype>
void FoldingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), channels_folded_, height_folded_, width_folded_);
  int count_ = bottom[0]->num() * bottom[0]->channels();
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, (*top)[0]->count());
  CHECK_EQ(bottom[0]->channels(), (*top)[0]->channels() * (*top)[0]->height() * (*top)[0]->width());
}

template <typename Dtype>
void FoldingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void FoldingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  (*bottom)[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(FoldingLayer);
#endif

INSTANTIATE_CLASS(FoldingLayer);

}  // namespace caffe
