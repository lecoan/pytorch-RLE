#include <torch/extension.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iterator>
#include <vector>
#include <string>
 


std::vector<torch::Tensor> rle_cuda_encode(torch::Tensor& input, int len) {
    const size_t N = input.numel();

    // allocate storage for output data and run lengths
    torch::Tensor output = torch::zeros(len, torch::device(torch::kCUDA).dtype(input.dtype()));
    torch::Tensor lengths = torch::zeros(len, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // compute run lengths
    auto len_ptr = (int32_t *)lengths.data_ptr();
    AT_DISPATCH_FLOATING_TYPES(input.type(), "reduce_by_key", (
        [&] {
            auto input_ptr = (scalar_t*)input.data_ptr();
            auto output_ptr = (scalar_t*)output.data_ptr();

            thrust::reduce_by_key
                    (thrust::device,
                        input_ptr, input_ptr+N,                  // input key sequence
                        thrust::constant_iterator<char>(1),      // input value sequence
                        output_ptr,                              // output key sequence
                        len_ptr                                  // output value sequence
                    );

        }
    ));
    return {output, lengths};
}

int rle_cuda_decode(torch::Tensor& input, torch::Tensor& lengths, torch::Tensor& output) {
    const size_t len = input.numel();
    // scan the lengths
    auto len_ptr = (int32_t*) lengths.data_ptr();
    thrust::inclusive_scan(thrust::device, len_ptr, len_ptr+len, len_ptr);

    // output size is sum of the run lengths
    int N = output.numel();

    // compute input index for each output element
    thrust::device_vector<int> indices(N);
    thrust::lower_bound(thrust::device, len_ptr, len_ptr+len,
                        thrust::counting_iterator<int>(1),
                        thrust::counting_iterator<int>(N + 1),
                        indices.begin());

    // gather input elements
    AT_DISPATCH_FLOATING_TYPES(output.type(), "gather", (
        [&] {
            auto input_ptr = (scalar_t*)input.data_ptr();
            auto output_ptr = (scalar_t*)output.data_ptr();
            thrust::gather(thrust::device, indices.begin(), indices.end(),
                            input_ptr, output_ptr);
        }
    ));
    return 0;
}