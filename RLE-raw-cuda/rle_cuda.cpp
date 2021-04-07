#include <torch/extension.h>

#include <vector>

#include <iostream>

// CUDA encode declarations

std::vector<at::Tensor> rle_cuda_encode(
    at::Tensor input, at::Tensor input_int);
    /*at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell*/

at::Tensor rle_cuda_decode(
    at::Tensor countsOut,
    at::Tensor symbolsOut,
    at::Tensor result);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> rle_encode(at::Tensor input, at::Tensor input_int) {
  CHECK_INPUT(input);
  //CHECK_INPUT(input_int);
  //std::cout << "in rle_cuda.cpp rle_encode"<< std::endl;
  return rle_cuda_encode(input, input_int);
}

at::Tensor rle_decode(
	at::Tensor countsOut,
	at::Tensor symbolsOut,
  at::Tensor result) {
  CHECK_INPUT(countsOut);
  CHECK_INPUT(symbolsOut);
  CHECK_INPUT(result);

  return rle_cuda_decode(
	  countsOut,
	  symbolsOut,
    result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encode", &rle_encode, "RLE encode (CUDA)");
  m.def("decode", &rle_decode, "RLE decode (CUDA)");
}