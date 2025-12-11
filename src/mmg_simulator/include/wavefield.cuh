#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

// 全局常量
constexpr float g = 9.81f;

namespace wavefield
{
    // =========================================================================
    // WaveFieldCalculator 类定义 (Host 接口)
    // =========================================================================

    class WaveFieldCalculator
    {
    private:
        // 波场和网格参数
        int Nx, Ny;
        float Lx_grid, Ly_grid;
        float dkx, dky;
        float Hs, Tp, gamma;
        unsigned long seed;

        int Ntot_complex;
        int Ntot_real;

        // Host 内存: 存储频率
        std::vector<float> h_omega;

        // Device 内存: 存储波谱 a(k) 和随机数状态
        cufftComplex *d_a_k = nullptr;
        curandState *d_states = nullptr;
        float *d_omega = nullptr;

    public:
        // 构造函数和析构函数
        WaveFieldCalculator(int nx, int ny, float lx, float ly, float hs, float tp, float g_factor, unsigned long s);
        ~WaveFieldCalculator();

        // ------------------- 全场查询接口 -------------------
        void query_wave_height(float t, float *h_out);
        void query_velocity_potential(float t, float *phi_out);

        float query_wave_height_point(float x, float y, float t);
        float query_velocity_potential_point(float x, float y, float t);

        // Accessors
        int get_Ntot_real() const { return Ntot_real; }

    private:
        // 辅助函数：将物理坐标 (x, y) 转换为数组索引
        size_t get_grid_index(float x, float y) const;
    };
}
