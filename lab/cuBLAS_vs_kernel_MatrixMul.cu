#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16  // 定义线程块大小

// 错误检查宏
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t err = call;                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << err << std::endl;                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 自定义矩阵乘法核函数，保持原状
__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;
    float Pvalue = 0;

    for (int ph = 0; ph < Width / BLOCK_SIZE; ++ph) {
        Mds[ty][tx] = d_M[Row * Width + ph * BLOCK_SIZE + tx];
        Nds[ty][tx] = d_N[(ph * BLOCK_SIZE + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}

int main() {
    int Width = 4096;  // 矩阵的维度，可以调整

    // 自定义核函数使用 float 类型
    size_t size_float = Width * Width * sizeof(float);

    // cuBLAS 使用 half 类型
    size_t size_half = Width * Width * sizeof(__half);

    // 分配主机内存
    float* h_A_float = (float*)malloc(size_float);
    float* h_B_float = (float*)malloc(size_float);
    float* h_C_float = (float*)malloc(size_float);      // 自定义核函数结果

    __half* h_A_half = (__half*)malloc(size_half);
    __half* h_B_half = (__half*)malloc(size_half);
    __half* h_C_half = (__half*)malloc(size_half);      // cuBLAS 结果

    // 初始化矩阵 A 和 B
    for (int i = 0; i < Width * Width; ++i) {
        float val = rand() / (float)RAND_MAX;
        h_A_float[i] = val;
        h_B_float[i] = val;
        h_A_half[i] = __float2half(val);
        h_B_half[i] = __float2half(val);
    }

    // 分配设备内存
    float *d_A_float, *d_B_float, *d_C_float;
    CHECK_CUDA(cudaMalloc((void**)&d_A_float, size_float));
    CHECK_CUDA(cudaMalloc((void**)&d_B_float, size_float));
    CHECK_CUDA(cudaMalloc((void**)&d_C_float, size_float));

    __half *d_A_half, *d_B_half, *d_C_half;
    CHECK_CUDA(cudaMalloc((void**)&d_A_half, size_half));
    CHECK_CUDA(cudaMalloc((void**)&d_B_half, size_half));
    CHECK_CUDA(cudaMalloc((void**)&d_C_half, size_half));

    // 将数据从主机复制到设备
    // 自定义核函数数据
    CHECK_CUDA(cudaMemcpy(d_A_float, h_A_float, size_float, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_float, h_B_float, size_float, cudaMemcpyHostToDevice));

    // cuBLAS 数据
    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half, size_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half, size_half, cudaMemcpyHostToDevice));

    // 计算线程块和网格维度
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(Width / dimBlock.x, Width / dimBlock.y);

    // 创建 CUDA 事件，用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float milliseconds = 0;

    // --------------------------
    // 1. 运行自定义核函数
    // --------------------------
    CHECK_CUDA(cudaEventRecord(start));
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A_float, d_B_float, d_C_float, Width);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "自定义核函数运行时间: " << milliseconds << " ms" << std::endl;

    // 将结果从设备复制回主机
    CHECK_CUDA(cudaMemcpy(h_C_float, d_C_float, size_float, cudaMemcpyDeviceToHost));

    // --------------------------
    // 2. 使用 cuBLAS 进行矩阵乘法（启用 Tensor Core）
    // --------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 设置数学模式以允许使用 Tensor Core
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 使用列主序，需要调整矩阵维度
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Width, Width, Width,
        &alpha,
        d_B_half, CUDA_R_16F, Width,
        d_A_half, CUDA_R_16F, Width,
        &beta,
        d_C_half, CUDA_R_16F, Width,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuBLAS（Tensor Core）运行时间: " << milliseconds << " ms" << std::endl;

    // 将 cuBLAS 结果复制回主机
    CHECK_CUDA(cudaMemcpy(h_C_half, d_C_half, size_half, cudaMemcpyDeviceToHost));

    // --------------------------
    // 3. 验证结果正确性
    // --------------------------
    float max_error = 0.0f;
    for (int i = 0; i < Width * Width; ++i) {
        float val_float = h_C_float[i];
        float val_half = __half2float(h_C_half[i]);
        float diff = fabs(val_float - val_half);
        if (diff > max_error) max_error = diff;
    }
    std::cout << "最大误差: " << max_error << std::endl;

    // 清理资源
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_float));
    CHECK_CUDA(cudaFree(d_B_float));
    CHECK_CUDA(cudaFree(d_C_float));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_C_half));
    free(h_A_float);
    free(h_B_float);
    free(h_C_float);
    free(h_A_half);
    free(h_B_half);
    free(h_C_half);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
