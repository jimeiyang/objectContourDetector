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
class UnpoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnpoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 2, 2, 2);
    blob_bottom_mask_->Reshape(2, 2, 2, 2);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_mask_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnpoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_mask_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_mask_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2 x 2 square unpooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
    unpooling_param->set_kernel_size(2);
    unpooling_param->set_stride(2);
    unpooling_param->set_unpool(UnpoolingParameter_UnpoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 2);
    blob_bottom_mask_->Reshape(num, channels, 2, 2);
    // Input data: 2x 2 channels of:
    //     [5 2]
    //     [9 4]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 5;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 9;
      blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    }
    // Input mask: 2x 2 channels of:
    //     [1   6]
    //     [13 15]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_mask_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_mask_->mutable_cpu_data()[i +  1] = 6;
      blob_bottom_mask_->mutable_cpu_data()[i +  2] = 13;
      blob_bottom_mask_->mutable_cpu_data()[i +  3] = 15;
    }
    UnpoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 4x4 channels of:
    //     [0 5 0 0]
    //     [0 0 2 0]
    //     [0 0 0 0]
    //     [0 9 0 4]
    for (int i = 0; i < 16 * num * channels; i += 16) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 4);
    }
  }
  // Test for 3x 2 rectangular unpooling layer with kernel_h > kernel_w
  void TestForwardRect() {
    LayerParameter layer_param;
    UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
    unpooling_param->set_kernel_h(3);
    unpooling_param->set_kernel_w(2);
    unpooling_param->set_stride_h(1);
    unpooling_param->set_stride_w(2);
    unpooling_param->set_unpool(UnpoolingParameter_UnpoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 2, 3);
    blob_bottom_mask_->Reshape(num, channels, 2, 3);
    // Input data: 2x 2 channels of:
    // [35    32    26]
    // [32    33    15]
    for (int i = 0; i < 6 * num * channels; i += 6) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 32;
      blob_bottom_->mutable_cpu_data()[i +  2] = 26;
      blob_bottom_->mutable_cpu_data()[i +  3] = 32;
      blob_bottom_->mutable_cpu_data()[i +  4] = 33;
      blob_bottom_->mutable_cpu_data()[i +  5] = 15;
    }
    // Input mask: 2x2 channels of: 
    // [ 0     3     5]
    // [ 7    20    22]
    for (int i = 0; i < 6 * num * channels; i += 6) {
      blob_bottom_mask_->mutable_cpu_data()[i +  0] =  0;
      blob_bottom_mask_->mutable_cpu_data()[i +  1] =  3;
      blob_bottom_mask_->mutable_cpu_data()[i +  2] =  5;
      blob_bottom_mask_->mutable_cpu_data()[i +  3] =  7;
      blob_bottom_mask_->mutable_cpu_data()[i +  4] = 20;
      blob_bottom_mask_->mutable_cpu_data()[i +  5] = 22;
    }
    UnpoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 6);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35     0     0    32     0    26]
    // [ 0    32     0     0     0     0]
    // [ 0     0     0     0     0     0]
    // [ 0     0    33     0    15     0]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21],  0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 15);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23],  0);
    }
  }
};

TYPED_TEST_CASE(UnpoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnpoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(UnpoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(2);
  unpooling_param->set_pad(1);
  unpooling_param->set_unpool(UnpoolingParameter_UnpoolMethod_MAX);
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(UnpoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
  this->TestForwardRect();
}

/*
TYPED_TEST(UnpoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 2; kernel_h <= 3; kernel_h++) {
    for (int kernel_w = 2; kernel_w <= 3; kernel_w++) {
      LayerParameter layer_param;
      UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_kernel_h(kernel_h);
      unpooling_param->set_kernel_w(kernel_w);
      unpooling_param->set_stride(2);
      unpooling_param->set_pad(1);
      unpooling_param->set_unpool(UnpoolingParameter_UnpoolMethod_MAX);
      UnpoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}
*/

TYPED_TEST(UnpoolingLayerTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(2);
  unpooling_param->set_stride(2);
  unpooling_param->set_pad(1);
  unpooling_param->set_unpool(UnpoolingParameter_UnpoolMethod_MAX);
  // set input data
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  // Input data:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0] = 1;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 4;
  this->blob_bottom_->mutable_cpu_data()[3] = 2;
  this->blob_bottom_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_->mutable_cpu_data()[5] = 2;
  this->blob_bottom_->mutable_cpu_data()[6] = 4;
  this->blob_bottom_->mutable_cpu_data()[7] = 2;
  this->blob_bottom_->mutable_cpu_data()[8] = 1;
  // set input mask
  this->blob_bottom_mask_->Reshape(1, 1, 3, 3);
  // Input data:
  //     [  0  2  3 ]
  //     [  4  9 11 ]
  //     [ 12 13 15 ]
  this->blob_bottom_mask_->mutable_cpu_data()[0] = 0;
  this->blob_bottom_mask_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_mask_->mutable_cpu_data()[2] = 3;
  this->blob_bottom_mask_->mutable_cpu_data()[3] = 4;
  this->blob_bottom_mask_->mutable_cpu_data()[4] = 9;
  this->blob_bottom_mask_->mutable_cpu_data()[5] = 11;
  this->blob_bottom_mask_->mutable_cpu_data()[6] = 12;
  this->blob_bottom_mask_->mutable_cpu_data()[7] = 13;
  this->blob_bottom_mask_->mutable_cpu_data()[8] = 15;
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Output:
  //     [ 1 0 2 4 ]
  //     [ 2 0 0 0 ]
  //     [ 0 3 0 2 ]
  //     [ 4 2 0 1 ]
  EXPECT_EQ(this->blob_top_->cpu_data()[0], 1);
  EXPECT_EQ(this->blob_top_->cpu_data()[1], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[2], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[3], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[4], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[5], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[6], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[7], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[8], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[9], 3);
  EXPECT_EQ(this->blob_top_->cpu_data()[10], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[11], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[12], 4);
  EXPECT_EQ(this->blob_top_->cpu_data()[13], 2);
  EXPECT_EQ(this->blob_top_->cpu_data()[14], 0);
  EXPECT_EQ(this->blob_top_->cpu_data()[15], 1);
}

/*
TYPED_TEST(UnpoolingLayerTest, TestForwardAve) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
  unpooling_param->set_kernel_size(3);
  unpooling_param->set_stride(1);
  unpooling_param->set_pad(1);
  unpooling_param->set_unpool(UnpoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Dtype epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}

TYPED_TEST(UnpoolingLayerTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_kernel_h(kernel_h);
      unpooling_param->set_kernel_w(kernel_w);
      unpooling_param->set_stride(2);
      unpooling_param->set_unpool(UnpoolingParameter_PoolMethod_AVE);
      UnpoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}

TYPED_TEST(UnpoolingLayerTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      UnpoolingParameter* unpooling_param = layer_param.mutable_unpooling_param();
      unpooling_param->set_kernel_h(kernel_h);
      unpooling_param->set_kernel_w(kernel_w);
      unpooling_param->set_stride(2);
      unpooling_param->set_pad(2);
      unpooling_param->set_unpool(UnpoolingParameter_PoolMethod_AVE);
      UnpoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
    }
  }
}
*/

}  // namespace caffe
