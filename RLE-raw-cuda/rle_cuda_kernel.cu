#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_vector.h>


// TOOD: where to use?
template <typename scalar_t>
__global__ void tempKernel(
    const scalar_t* __restrict__ g_in, 
    scalar_t* __restrict__ g_temp,  
    size_t n) { 
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	
    for (int i = index; i < n; i += stride) {
        if (g_in[i]-0.0 < 0.0001 && g_in[i]-0.0 > -0.0001){
            g_temp[i] = 0;
        }
        else {
            g_temp[i] = 2;
        }
    }
} 

template <typename scalar_t>
__global__ void maskKernel(
    const scalar_t* __restrict__ g_in, 
    int* __restrict__ g_decodeMask,
    size_t n) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	
    for (int i = index; i < n; i += stride) {
        if (g_in[i] == 0){
            g_decodeMask[i] = 0;
        }
        else {
            g_decodeMask[i] = 1;
        }

    }
} 

// TODO: where to use?
__global__ void prefixsumKernel(
    const int* __restrict__ X, 
    int* __restrict__ XY,
    int* __restrict__ Y, 
    size_t InputSize) {
    auto BLOCK_SIZE = 32*((InputSize+32)/32);
    //printf("BLOCK_SIZE=%d\n", BLOCK_SIZE);
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < InputSize) {XY[threadIdx.x] = X[i];}

    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x+1)*stride*2 - 1; 
        if(index < 2*BLOCK_SIZE)
            XY[index] += XY[index - stride];  //index is alway bigger than stride
    }
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
    //for (int stride2 = BLOCK_SIZE/2; stride2 > 0; stride2 = stride2/2) {
        int index2 = (threadIdx.x+1)*stride*2 - 1;
        if(index2 < 2*BLOCK_SIZE)
            XY[index2 + stride] += XY[index2];
    }
    if (i < InputSize) Y[i] = XY[threadIdx.x];
}  

 
__global__ void compactKernel(int* __restrict__ g_scannedBackwardMask,
                              int* g_compactedBackwardMask,
                              int* g_totalRuns,
                              size_t n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	
    for (int i = index; i < n; i += stride) {
        if (i == (n - 1)) {
            g_compactedBackwardMask[g_scannedBackwardMask[i]] = i + 1;
            *g_totalRuns = g_scannedBackwardMask[i];
        }

        if (i == 0) {
            if(g_scannedBackwardMask[0] == 1) {
                g_compactedBackwardMask[0] = 0;
            }
        }
        else if (g_scannedBackwardMask[i] != g_scannedBackwardMask[i - 1]) {
                g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i;
        }
        g_compactedBackwardMask[g_scannedBackwardMask[n-1]] = n;
        g_totalRuns[0] = g_scannedBackwardMask[n-1];
    }
}


template <typename scalar_t>
__global__ void scatterKernel(
          int* g_compactedBackwardMask,
          int* g_totalRuns,
          scalar_t* __restrict__ g_countsOut) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	
    int n = *g_totalRuns;
    for (int i = index; i < n; i += stride) {
        if (i == 0)
            g_countsOut[i] = g_compactedBackwardMask[i];
        else
            g_countsOut[i] = g_compactedBackwardMask[i] - g_compactedBackwardMask[i-1] - 1;
            
    }	
    g_countsOut[n] = g_compactedBackwardMask[n];
}   

template <typename scalar_t>
__global__ void recordKernel(
          int* g_compactedBackwardMask,
          int* g_totalRuns,
          scalar_t* __restrict__ g_in,
          scalar_t* __restrict__ g_symbolsOut) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	
    int n = *g_totalRuns;

    for (int i = index; i < n; i += stride) {
        if(g_compactedBackwardMask[i] != -1){
            g_symbolsOut[i] = g_in[g_compactedBackwardMask[i]];
        }
    }
}  

// TODO: where to use?
std::vector<at::Tensor> rle_cuda_encode_2(at::Tensor input, at::Tensor input_int) {
    const auto n = input.size(1);
    const int threads = 512;    //256
    int blocks = (n + threads ) / threads;
    if(blocks > 65535)
        blocks = 65535;
    int *compactedBackwardMask;
    auto g_countsOut = at::ones({1, n}, input_int.type()).to(at::kCUDA);
    auto g_symbolsOut = at::ones({1, n}, input.type()).to(at::kCUDA);

    if(0 != cudaMalloc(&compactedBackwardMask, n*sizeof(int)))
        std::cout<<__LINE__<<"  cudaMalloc error "<<std::endl;
    thrust::inclusive_scan(thrust::device, compactedBackwardMask, compactedBackwardMask + n, compactedBackwardMask);

    return {g_countsOut, g_symbolsOut};
}

std::vector<at::Tensor> rle_cuda_encode(at::Tensor input, at::Tensor input_int) {
    int device;	
    cudaGetDevice(&device);
    const auto n = input.size(1);

    const int threads = 512;    //256
    int blocks = (n + threads ) / threads;
    if(blocks > 65535)
        blocks = 65535;

    int *decodeMask, *scannedBackwardMask;
    if(0 != cudaMalloc(&decodeMask, n*sizeof(int)))
        std::cout<<__LINE__<<"  cudaMalloc error "<<std::endl;

    cudaDeviceSynchronize();
    if(0 != cudaMalloc(&scannedBackwardMask, n*sizeof(int)))
        std::cout<<__LINE__<<"  cudaMalloc error "<<std::endl;
    auto err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    AT_DISPATCH_INTEGRAL_TYPES(input_int.type(), "rle_encode_cuda", ([&] {
      maskKernel<scalar_t><<<blocks, threads>>>(
        input_int.data<scalar_t>(),
        decodeMask,
        n);
    }));
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    thrust::inclusive_scan(thrust::device, decodeMask, decodeMask + n, scannedBackwardMask);
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaFree(decodeMask);
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    cudaDeviceSynchronize();
    int *totalRuns, *compactedBackwardMask;

    cudaMalloc(&compactedBackwardMask, (n+1)*sizeof(int));
    cudaDeviceSynchronize();
    cudaMallocManaged(&totalRuns, sizeof(int));
    cudaDeviceSynchronize();
    compactKernel<<<blocks, threads>>>(scannedBackwardMask, compactedBackwardMask, totalRuns, n);
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaFree(scannedBackwardMask);
    cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    int k = totalRuns[0]+1;
    auto g_countsOut = at::ones({1, k}, input_int.type()).to(at::kCUDA);
    cudaDeviceSynchronize();

    AT_DISPATCH_INTEGRAL_TYPES(input_int.type(), "rle_encode_cuda", ([&] {
        scatterKernel<scalar_t><<<blocks, threads>>>(
            compactedBackwardMask,
            totalRuns,
            g_countsOut.data<scalar_t>()); 
    }));
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    auto g_symbolsOut = at::ones({1, *totalRuns}, input.type()).to(at::kCUDA);
    cudaDeviceSynchronize();
    AT_DISPATCH_FLOATING_TYPES(input.type(), "rle_encode_cuda", ([&] {
        recordKernel<scalar_t><<<blocks, threads>>>(
            compactedBackwardMask,
            totalRuns,
            input.data<scalar_t>(),
            g_symbolsOut.data<scalar_t>()); 
    }));
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaFree(compactedBackwardMask);
    cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaFree(totalRuns);
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    return {g_countsOut, g_symbolsOut};
}

template <typename scalar_t>
__global__ void sumzeroKernel(
          scalar_t* __restrict__ g_countsOut,
          int* result,
          size_t n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	

    for (int i = index; i < n; i += stride) {
        result[i] = g_countsOut[i];
    }
}  

__global__ void sumindexKernel(
          int* result,
          size_t n ) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;	

    for (int i = index; i < n; i += stride) {
        result[i] += i;
    }
}  

template <typename scalar_t>
__global__ void decodeKernel(
            int* temp,
            scalar_t* __restrict__ g_symbolsOut,
            scalar_t* __restrict__ g_output,
            size_t n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        g_output[temp[i]] = g_symbolsOut[i];
    }
}


at::Tensor rle_cuda_decode(at::Tensor countsOut, at::Tensor symbolsOut, at::Tensor result) {
    const auto n = symbolsOut.size(1);
    const int threads = 256;    //256
    int blocks = (n + threads - 1) / threads;
    if(blocks > 65535)
        blocks = 65535;
    int *temp;
    cudaError err;
    if(0 != cudaMalloc(&temp, n*sizeof(int)))
        std::cout<<__LINE__<<"  malloc failed"<<std::endl;

     err = cudaDeviceSynchronize();
     if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    AT_DISPATCH_INTEGRAL_TYPES(countsOut.type(), "rle_encode_cuda", ([&] {
        sumzeroKernel<scalar_t><<<blocks, threads>>>(
        countsOut.data<scalar_t>(),
        temp,
        n);
    }));
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    thrust::inclusive_scan(thrust::device, temp, temp + n, temp);
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    sumindexKernel<<<blocks, threads>>>(temp, n);

    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    auto totalSize = symbolsOut.size(1);
    cudaDeviceSynchronize();
    AT_DISPATCH_FLOATING_TYPES(symbolsOut.type(), "rle_encode_cuda", ([&] {
        decodeKernel<scalar_t><<<blocks, threads>>>(
        temp,
        symbolsOut.data<scalar_t>(),
        result.data<scalar_t>(),
        totalSize);
    }));
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaFree(temp);
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    return result;
}

