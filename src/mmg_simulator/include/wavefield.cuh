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

        // Host 端数据
        std::vector<float> h_omega;
        std::vector<float> h_k_abs;
        std::vector<cufftComplex> h_a_k;

        // Device 端数据
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
        const std::vector<float> &get_h_omega() const { return h_omega; }
        const std::vector<float> &get_h_k_abs() const { return h_k_abs; }
        const std::vector<cufftComplex> &get_h_a_k() const { return h_a_k; } // 关键：Host端波幅
        int get_Nx() const { return Nx; }
        int get_Ny() const { return Ny; }
        float get_Lx() const { return Lx_grid; }
        float get_Ly() const { return Ly_grid; }
        float get_dkx() const { return dkx; }
        float get_dky() const { return dky; }

    private:
        // 辅助函数：将物理坐标 (x, y) 转换为数组索引
        size_t get_grid_index(float x, float y) const;
    };
}
