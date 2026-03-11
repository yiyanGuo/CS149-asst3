#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
#define ELEMENT_PER_BLOCK (2 * THREADS_PER_BLOCK)
#define OFFSET(idx) idx + (idx >> 5)
// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
__global__ void exclusive_scan_block(int *result, int *block_sums, int N) {
    extern __shared__ int temp[]; 

    int t = threadIdx.x;
    int bid = blockIdx.x;
    size_t global_offset = bid * ELEMENT_PER_BLOCK;
    size_t shared_offset = t;

    // 1. 安全加载
    temp[OFFSET(shared_offset)] = (global_offset + shared_offset < N) ? result[global_offset + shared_offset] : 0;
    temp[OFFSET(shared_offset + THREADS_PER_BLOCK)] = (global_offset + shared_offset + THREADS_PER_BLOCK < N) ? result[global_offset + shared_offset + THREADS_PER_BLOCK] : 0;
    __syncthreads();

    // 2. Up-sweep (Reduce)
    for (int d = 1; d <= THREADS_PER_BLOCK; d *= 2) {
        int dd = 2 * d;
        if (t < (ELEMENT_PER_BLOCK / dd)) {
            int idx = (t + 1) * dd - 1;
            temp[OFFSET(idx)] += temp[OFFSET(idx - d)];
        }
        __syncthreads();
    }

    // 3. 提取总和并清零末位
    if (t == 0) {
        if (block_sums != nullptr) block_sums[bid] = temp[OFFSET(ELEMENT_PER_BLOCK - 1)];
        temp[OFFSET(ELEMENT_PER_BLOCK - 1)] = 0;
    }
    __syncthreads();

    // 4. Down-sweep
    for (int d = THREADS_PER_BLOCK; d >= 1; d /= 2) {
        int dd = 2 * d;
        if (t < (ELEMENT_PER_BLOCK / dd)) {
            int idx = (t + 1) * dd - 1;
            int val = temp[OFFSET(idx - d)];
            temp[OFFSET(idx - d)] = temp[OFFSET(idx)];
            temp[OFFSET(idx)] += val;
        }
        __syncthreads();
    }

    // 5. 安全写回
    if (global_offset + shared_offset < N) 
        result[global_offset + shared_offset] = temp[OFFSET(shared_offset)];
    if (global_offset + shared_offset + THREADS_PER_BLOCK < N) 
        result[global_offset + shared_offset + THREADS_PER_BLOCK] = temp[OFFSET(shared_offset + THREADS_PER_BLOCK)];
}

__global__ void add_prefix_sum(int *result, int *block_sums, int N) {
    size_t i1 = (size_t)blockIdx.x * ELEMENT_PER_BLOCK + threadIdx.x;
    size_t i2 = i1 + THREADS_PER_BLOCK;
    
    int pre_block_sum = block_sums[blockIdx.x];

    if (i1 < N) result[i1] += pre_block_sum;
    if (i2 < N) result[i2] += pre_block_sum;
}


void exclusive_scan_impl(int* result, int *block_sums,int N)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    int num_blocks = (N + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK;
    
    if(N <= ELEMENT_PER_BLOCK) {
        // 直接在一个块内完成扫描
        size_t shared_mem_size = ELEMENT_PER_BLOCK * sizeof(int);
        shared_mem_size += shared_mem_size / 32; // 预留一些共享内存用于避免bank conflict
        exclusive_scan_block<<<1, THREADS_PER_BLOCK, shared_mem_size>>>(result, nullptr, N);
        return;
    }

    // A: block sums
    size_t shared_mem_size = ELEMENT_PER_BLOCK * sizeof(int);
    shared_mem_size += shared_mem_size / 32; // 预留一些共享内存用于避免bank conflict
    exclusive_scan_block<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(result, block_sums, N);

    // B: scan block sums
    exclusive_scan_impl(block_sums, block_sums + num_blocks, num_blocks);

    // C: add base
    add_prefix_sum<<<num_blocks, THREADS_PER_BLOCK>>>(result, block_sums, N);
}

void exclusive_scan(int *input, int N, int *result) {
    N = nextPow2(N);
    int *device_block_sums;
    size_t block_sums_size = 0;
    for(size_t n = N / ELEMENT_PER_BLOCK; n > 1; n = (n + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK) {
        block_sums_size += sizeof(int) * n;
    }
    cudaMalloc(&device_block_sums, block_sums_size);
    exclusive_scan_impl(result, device_block_sums, N);

    cudaFree(device_block_sums);
}
//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
__global__ void map_repeats(int *input, int *output, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N - 1) {
        output[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
    }
    else if (idx == N - 1) {
        output[idx] = 0; // 最后一个元素没有下一个元素，不能构成重复对
    }
}

__global__ void extract_repeats(int *input, int input_length, int *output, int output_length) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_length - 1) {
        if (input[idx] < input[idx + 1]) {
            int output_idx = input[idx];
            if (output_idx < output_length) {
                output[output_idx] = idx; // 将重复对的起始索引写入输出数组
            }
        }
    }
}

int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    map_repeats<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, device_output, length);
    cudaDeviceSynchronize();

    exclusive_scan(device_output, length, device_output);

    int output_length;
    cudaMemcpy(&output_length, device_output + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    extract_repeats<<<numBlocks, THREADS_PER_BLOCK>>>(device_output, length, device_output, output_length);
    cudaDeviceSynchronize();
    return output_length; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
