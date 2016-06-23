// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/local_update.hpp"

namespace caffe {

template <typename Dtype>
__global__ void local_update1_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                    Dtype* data_R, const int input_num,
                                    const int location_num, const int output_num) {
  int total = input_num * location_num * output_num;
  CUDA_KERNEL_LOOP(index, total) {
    int n = index % location_num; //location index
    int p = (index / location_num) % input_num; //input index
    int q = (index / location_num) / input_num; //output index
    data_R[index] += data_A[q*location_num+n] * data_B[p*location_num+n];
  }
}

template <typename Dtype>
void local_update1_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int input_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is input_num x location_num
  // data_R is output_num x input_num x location_num, the update performed is Rqpn += Aqn * Bpn

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(input_num * location_num * output_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, input_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update1_gpu<float>(const float* data_A, const float* data_B,
                                float* data_R, const int input_num,
                                const int location_num, const int output_num);
template void local_update1_gpu<double>(const double* data_A, const double* data_B,
                                double* data_R, const int input_num,
                                const int location_num, const int output_num);


template <typename Dtype>
__global__ void local_update2_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                Dtype* data_R, const int input_num,
                                const int location_num, const int output_num) {
  int total = input_num * location_num;
  CUDA_KERNEL_LOOP(index, total) {
    int n = index % location_num; //location index
    int p = (index / location_num); //input index
    for (int q=0; q<output_num; q++) {
      data_R[index] += data_A[q*location_num+n] * data_B[(q*input_num+p)*location_num+n];
    }
  }
}

template <typename Dtype>
void local_update2_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int input_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is output_num x input_num x location_num
  // data_R is input_num x location_num, the update performed is Rpn += \sum_q(Aqn * Bqpn)

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(input_num * location_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, input_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update2_gpu<float>(const float* data_A, const float* data_B,
                       float* data_R, const int input_num,
                       const int location_num, const int output_num);
template void local_update2_gpu<double>(const double* data_A, const double* data_B,
                       double* data_R, const int input_num,
                       const int location_num, const int output_num);


template <typename Dtype>
__global__ void local_update3_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                Dtype* data_R, const int input_num,
                                const int location_num, const int output_num) {
  int total = output_num * location_num;
  CUDA_KERNEL_LOOP(index, total) {
    int n = index % location_num; //location index
    int q = (index / location_num); //output index
    for (int p=0; p<input_num; p++) {
      data_R[index] += data_A[p*location_num+n] * data_B[(q*input_num+p)*location_num+n];
    }
  }
}

template <typename Dtype>
void local_update3_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int input_num,
                       const int location_num, const int output_num) {
  // data_A is input_num x location_num
  // data_B is input_num x input_num x location_num
  // data_R is output_num x location_num, the update performed is Rqn += \sum_p(Apn * Bqpn)

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update3_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(output_num * location_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, input_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update3_gpu<float>(const float* data_A, const float* data_B,
                       float* data_R, const int input_num,
                       const int location_num, const int output_num);
template void local_update3_gpu<double>(const double* data_A, const double* data_B,
                       double* data_R, const int input_num,
                       const int location_num, const int output_num);


}  // namespace caffe
