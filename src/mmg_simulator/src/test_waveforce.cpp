#include "extwave.hpp"
#include "wavefield.cuh"
#include <iostream>
#include <cmath>
#include <tuple>
#include <iomanip>

int main()
{
    std::cout << "--- 初始化 Froude–Krylov 波浪力计算模块 ---\n";

    // ============================================================
    // 1. 波场与网格参数
    // ============================================================
    const int Nx = 64;
    const int Ny = 64;
    const float Lx = 100.0f;
    const float Ly = 100.0f;

    // JONSWAP 参数
    const float Hs = 0.5f;
    const float Tp = 3.5f;
    const float gamma = 3.3f;
    const unsigned long seed = 42;

    // ============================================================
    // 2. 初始化 WaveFieldCalculator 与 waveforce_generator
    // ============================================================
    try
    {
        std::cout << "创建波场计算器 WaveFieldCalculator...\n";
        wavefield::WaveFieldCalculator wave_calc(
            Nx, Ny, Lx, Ly, Hs, Tp, gamma, seed);
        std::cout << "WaveFieldCalculator 初始化成功。\n";

        std::cout << "创建波浪力计算器 WaveForceGenerator...\n";
        waveforce::WaveForceCalculator generator(wave_calc,
                                                 1.3f,    // 船长 L (m)
                                                 0.98f,   // 船宽 B (m)
                                                 0.12f,   // 吃水 draft (m)
                                                 1025.0f, // 水密度 rho (kg/m^3)
                                                 8,       // 横向采样数 n_span
                                                 8        // 竖向采样数 n_vert
        );
        std::cout << "WaveForceGenerator 初始化成功，船体面元已构建。\n";

        // ============================================================
        // 3. 仿真参数
        // ============================================================
        const float T_total = 20.0f;
        const float dt = 0.1f;

        float x_ship = 0.0f;
        float y_ship = 0.0f;
        float psi = 0.0f;

        // ============================================================
        // 4. 输出表头
        // ============================================================
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n  t(s) |    x(m)   |    y(m)   | psi(rad) |    Fu(N)    |    Fv(N)    |   Mr(Nm)\n";
        std::cout << "--------------------------------------------------------------------------------------------\n";

        // ============================================================
        // 5. 主时间循环
        // ============================================================
        for (float t = 0.0f; t <= T_total; t += dt)
        {

            // 示例运动：半径 5m 的螺旋轨迹
            x_ship = 5.0f * std::cos(t * 0.1f);
            y_ship = 5.0f * std::sin(t * 0.1f);
            psi = 0.1f * t;

            // 调用 Froude–Krylov 力计算器
            auto Force = generator.compute_force(t, x_ship, y_ship, psi);

            // 打印输出
            std::cout
                << std::setw(6) << t << " | "
                << std::setw(9) << x_ship << " | "
                << std::setw(9) << y_ship << " | "
                << std::setw(9) << psi << " | "
                << std::setw(10) << Force.Fx << " | "
                << std::setw(10) << Force.Fy << " | "
                << std::setw(10) << Force.Mz << "\n";
        }
    }

    // ============================================================
    // 6. 异常捕获
    // ============================================================
    catch (const std::exception &e)
    {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "发生未知异常。" << std::endl;
        return 1;
    }

    return 0;
}
