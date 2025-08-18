#ifndef _EXTWAVE_CUH_
#define _EXTWAVE_CUH_

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <curand_kernel.h>
#include <vector>

class CudaWaveForceGenerator {
public:
    // 构造参数：N采样点数，dt采样周期，Hs波高，Tp峰值周期，waveDirectionRad波浪方向
    CudaWaveForceGenerator(int N_, float dt_, float Hs, float Tp, float waveDirectionRad, float L, float B);
    ~CudaWaveForceGenerator();

    // 获取t时刻全局波浪力（Fx,Fy,Mz）
    Eigen::Vector3f getWaveForceGlobal(float t);

    // 根据state投影波浪力到船体坐标系τ1,τ2,τ6
    __host__ Eigen::Vector3f getWaveForce(const Eigen::Matrix<float,6,1>& state, float t);

private:
    int N;
    float dt, T;
    float wave_dir;
    float Hs, L, B;

    thrust::complex<float>* d_spectrum_fx;
    thrust::complex<float>* d_spectrum_fy;
    thrust::complex<float>* d_spectrum_mz;

    float* d_time_series; // IFFT后时域信号：N点，每点3个浮点

    void allocateCudaMemory();
    void freeCudaMemory();
    void runIFFT();
};

#endif // _EXTWAVE_CUH_
