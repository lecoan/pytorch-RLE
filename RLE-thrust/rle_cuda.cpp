#include <torch/extension.h>

#include <vector>
 
// CUDA encode declarations
std::vector<torch::Tensor> rle_cuda_encode(torch::Tensor& input, int len);

int rle_cuda_decode(
    torch::Tensor& countsOut,
    torch::Tensor& symbolsOut,
    torch::Tensor& result);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 设非零元素为k, 返回长度为两个列表
// 长度为k的浮点数
// 长度为k+1
std::vector<torch::Tensor> rle_encode(torch::Tensor input, int len) {
  CHECK_INPUT(input);
  return rle_cuda_encode(input, len);
}

// 结果放在result
int rle_decode(
	torch::Tensor& countsOut,
	torch::Tensor& symbolsOut,
  torch::Tensor& result) {
  CHECK_INPUT(countsOut);
  CHECK_INPUT(symbolsOut);
  CHECK_INPUT(result);

  return rle_cuda_decode(
    countsOut,
    symbolsOut,
    result);
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encode", &rle_encode, "RLE encode (CUDA)");
  m.def("decode", &rle_decode, "RLE decode (CUDA)");
}