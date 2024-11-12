#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

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

int main() {
    // 矩阵和向量的尺寸
    int M = 4096; // 向量长度
    int N = 4096; // 矩阵列数

    // 分配主机内存
    size_t size_vector = M * sizeof(__half);
    size_t size_matrix = M * N * sizeof(__half);
    size_t size_result = N * sizeof(__half);

    __half* h_x = (__half*)malloc(size_vector);   // 向量 x
    __half* h_A = (__half*)malloc(size_matrix);   // 矩阵 A
    __half* h_y = (__half*)malloc(size_result);   // 结果向量 y

    // 初始化向量 x 和矩阵 A
    for (int i = 0; i < M; ++i) {
        h_x[i] = __float2half(1.0f); // 或者使用随机值
    }
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = __float2half(1.0f); // 或者使用随机值
    }

    // 分配设备内存
    __half* d_x;
    __half* d_A;
    __half* d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, size_vector));
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_matrix));
    CHECK_CUDA(cudaMalloc((void**)&d_y, size_result));

    // 将数据从主机复制到设备
    CHECK_CUDA(cudaMemcpy(d_x, h_x, size_vector, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_matrix, cudaMemcpyHostToDevice));

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 设置数学模式以允许使用 Tensor Core
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 设置 CUDA 事件，用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float milliseconds = 0.0f;

    // 执行向量乘以矩阵操作：y = alpha * x * A + beta * y
    // 在 cuBLAS 中，向量和矩阵需要以列主序存储，因此需要调整参数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 开始计时
    CHECK_CUDA(cudaEventRecord(start));

    // 使用 cublasGemmEx 实现 y = x * A
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, 1, M,
        &alpha,
        d_A, CUDA_R_16F, N,
        d_x, CUDA_R_16F, M,
        &beta,
        d_y, CUDA_R_16F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // 停止计时
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "运行时间: " << milliseconds << " ms" << std::endl;

    // 将结果从设备复制回主机
    CHECK_CUDA(cudaMemcpy(h_y, d_y, size_result, cudaMemcpyDeviceToHost));

    // 验证结果（简单验证）
    // 由于 x 和 A 的元素都是 1，预期 y 的每个元素都是 M
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float y_value = __half2float(h_y[i]);
        if (fabs(y_value - M) > 1e-2) {
            correct = false;
            std::cerr << "结果错误，索引 " << i << "，值: " << y_value << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "结果正确！" << std::endl;
    } else {
        std::cout << "结果错误！" << std::endl;
    }

    // 清理资源
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_A);
    free(h_y);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
