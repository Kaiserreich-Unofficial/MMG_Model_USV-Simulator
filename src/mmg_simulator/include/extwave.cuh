#ifndef _EXTWAVE_CUH_
#define _EXTWAVE_CUH_

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

// CUDA runtime 错误检查
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUFFT 错误检查
#define CUFFT_CHECK(err) \
    do { \
        cufftResult err_ = (err); \
        if (err_ != CUFFT_SUCCESS) { \
            fprintf(stderr, "CUFFT Error: %d (%s:%d)\n", err_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class CudaWaveForceGenerator
{
public:
    CudaWaveForceGenerator(
        int N, float dt, float Hs, float Tp, float waveDirRad,
        float L, float B, float draft, float waterDepth
    );
    ~CudaWaveForceGenerator();

    // 返回全局坐标系 u,v,udot,vdot
    Eigen::Vector4f getWaveKinematicsGlobal(float t);

    // 返回船体坐标系波浪力 [Fx,Fy,N]
    Eigen::Vector3f getWaveForce(const Eigen::Matrix<float,6,1> &state, float t);

private:
    void allocateCudaMemory();
    void freeCudaMemory();
    void runIFFTAndStoreTimeSeries();

    int N;
    float dt, Hs, Tp, wave_dir;
    float L,B,draft,water_depth;
    float T; // 总模拟时间

    // Morison 参数
    float Cd_x, Cd_y;   // 阻力系数
    float A_x, A_y;     // 投影面积
    float Cm;           // 惯性系数
    float V;            // 排水体积

    // RAO 系数 (频率依赖)
    std::vector<float> freq_vals;
    std::vector<float> RAO_surge;
    std::vector<float> RAO_sway;

    // IFFT 结果存储 u,v,udot,vdot
    float *d_time_series = nullptr; // device, length N*4

    // 频域波谱
    thrust::complex<float> *d_spectrum_ux = nullptr;
    thrust::complex<float> *d_spectrum_uy = nullptr;
    thrust::complex<float> *d_spectrum_ax = nullptr;
    thrust::complex<float> *d_spectrum_ay = nullptr;
};

#endif //_EXTWAVE_CUH_
