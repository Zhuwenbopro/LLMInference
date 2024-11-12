#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>

__global__ void silu_fp16_kernel(const __half* input, __half* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        __half x_half = input[idx];
        // 将 __half 转换为 float 进行计算
        float x = __half2float(x_half);
        // 计算 sigmoid(x) = 1 / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + expf(-x));
        // 计算 SiLU(x) = x * sigmoid(x)
        float silu = x * sigmoid;
        // 将结果转换回 __half
        output[idx] = __float2half(silu);
    }
}


__global__ void silu_fp32_kernel(const float* input, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // 计算 sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-x));
        // 计算 x * sigmoid(x)
        output[idx] = x * sigmoid;
    }
}

int main() {
    // 定义数据大小
    const int size = 1 << 24; // 大约 16M 个元素
    const int bytes_fp16 = size * sizeof(__half);
    const int bytes_fp32 = size * sizeof(float);

    // 分配主机内存
    float* h_input_fp32 = new float[size];
    float* h_output_fp32 = new float[size];
    float* h_output_fp16 = new float[size];

    // 初始化输入数据
    for (int i = 0; i < size; ++i) {
        h_input_fp32[i] = static_cast<float>(rand()) / RAND_MAX; // 随机数在 0 到 1 之间
    }

    // 分配设备内存
    float* d_input_fp32;
    float* d_output_fp32;
    __half* d_input_fp16;
    __half* d_output_fp16;

    cudaMalloc(&d_input_fp32, bytes_fp32);
    cudaMalloc(&d_output_fp32, bytes_fp32);
    cudaMalloc(&d_input_fp16, bytes_fp16);
    cudaMalloc(&d_output_fp16, bytes_fp16);

    // 将输入数据复制到设备（FP32）
    cudaMemcpy(d_input_fp32, h_input_fp32, bytes_fp32, cudaMemcpyHostToDevice);

    // 将 FP32 输入数据转换为 FP16 并复制到设备
    __half* h_input_fp16 = new __half[size];
    for (int i = 0; i < size; ++i) {
        h_input_fp16[i] = __float2half(h_input_fp32[i]);
    }
    cudaMemcpy(d_input_fp16, h_input_fp16, bytes_fp16, cudaMemcpyHostToDevice);

    // 定义 CUDA 内核执行配置
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // 计时变量
    cudaEvent_t start_fp32, stop_fp32;
    cudaEvent_t start_fp16, stop_fp16;
    cudaEventCreate(&start_fp32);
    cudaEventCreate(&stop_fp32);
    cudaEventCreate(&start_fp16);
    cudaEventCreate(&stop_fp16);

    // 运行并计时 FP32 核函数
    cudaEventRecord(start_fp32);
    silu_fp32_kernel<<<blocks, threads>>>(d_input_fp32, d_output_fp32, size);
    cudaEventRecord(stop_fp32);

    // 运行并计时 FP16 核函数
    cudaEventRecord(start_fp16);
    silu_fp16_kernel<<<blocks, threads>>>(d_input_fp16, d_output_fp16, size);
    cudaEventRecord(stop_fp16);

    // 等待计算完成
    cudaDeviceSynchronize();

    // 计算 FP32 运行时间
    float time_fp32 = 0;
    cudaEventElapsedTime(&time_fp32, start_fp32, stop_fp32);

    // 计算 FP16 运行时间
    float time_fp16 = 0;
    cudaEventElapsedTime(&time_fp16, start_fp16, stop_fp16);

    // 将结果复制回主机（FP32）
    cudaMemcpy(h_output_fp32, d_output_fp32, bytes_fp32, cudaMemcpyDeviceToHost);

    // 将结果复制回主机（FP16），并转换为 FP32 以便比较
    cudaMemcpy(h_input_fp16, d_output_fp16, bytes_fp16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        h_output_fp16[i] = __half2float(h_input_fp16[i]);
    }

    // 输出运行时间
    std::cout << "FP32 kernel execution time: " << time_fp32 << " ms" << std::endl;
    std::cout << "FP16 kernel execution time: " << time_fp16 << " ms" << std::endl;

    // 验证结果的正确性（可选）
    int error_count = 0;
    for (int i = 0; i < size; ++i) {
        float diff = fabs(h_output_fp32[i] - h_output_fp16[i]);
        if (diff > 1e-2) { // 允许一定的误差
            error_count++;
        }
    }
    if (error_count == 0) {
        std::cout << "Results are correct within acceptable error margin." << std::endl;
    } else {
        std::cout << "There are " << error_count << " mismatches between FP32 and FP16 results." << std::endl;
    }

    // 释放资源
    delete[] h_input_fp32;
    delete[] h_input_fp16;
    delete[] h_output_fp32;
    delete[] h_output_fp16;
    cudaFree(d_input_fp32);
    cudaFree(d_output_fp32);
    cudaFree(d_input_fp16);
    cudaFree(d_output_fp16);
    cudaEventDestroy(start_fp32);
    cudaEventDestroy(stop_fp32);
    cudaEventDestroy(start_fp16);
    cudaEventDestroy(stop_fp16);

    return 0;
}
