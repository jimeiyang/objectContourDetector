#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UpsamplingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UpsamplingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 2, 2, 2);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UpsamplingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square upsampling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
    upsampling_param->set_kernel_size(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 2);
    // Input: 2x 2 channels of:
    //     [1 2]
    //     [4 5]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 4;
      blob_bottom_->mutable_cpu_data()[i +  3] = 5;
    }
    UpsamplingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [1 0 2 0]
    //     [0 0 0 0]
    //     [4 0 5 0]
    //     [0 0 0 0]
    for (int i = 0; i < 16 * num * channels; i += 16) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
    }
  }
  // Test for 3x 2 rectangular upsampling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
    upsampling_param->set_kernel_h(3);
    upsampling_param->set_kernel_w(2);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 2);
    // Input: 2x 2 channels of:
    // [35     1]
    // [ 3    32]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 3;
      blob_bottom_->mutable_cpu_data()[i +  3] = 32;
    }
    UpsamplingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    0     1     0]
    // [0     0     0     0]
    // [0     0     0     0]
    // [3     0     32    0]
    // [0     0     0     0]
    // [0     0     0     0]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 0);
    }
  }
  // Test for rectangular upsampling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
    upsampling_param->set_kernel_h(2);
    upsampling_param->set_kernel_w(3);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 2);
    // Input: 2x 2 channels of:
    // [35     1]
    // [ 3    32]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 3;
      blob_bottom_->mutable_cpu_data()[i +  3] = 32;
    }
    UpsamplingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 6);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    0     0     1    0    0]
    // [0     0     0     0    0    0]
    // [3     0     0     32   0    0]
    // [0     0     0     0    0    0]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 0);
    }
  }
};

TYPED_TEST_CASE(UpsamplingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpsamplingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
  upsampling_param->set_kernel_size(2);
  UpsamplingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
}


TYPED_TEST(UpsamplingLayerTest, TestForward) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
}

TYPED_TEST(UpsamplingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
      upsampling_param->set_kernel_h(kernel_h);
      upsampling_param->set_kernel_w(kernel_w);
      UpsamplingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

}  // namespace caffe
