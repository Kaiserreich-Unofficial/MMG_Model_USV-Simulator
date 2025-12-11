#include "wavefield.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "matplotlibcpp.h"  // 需要先安装 matplotlibcpp

namespace plt = matplotlibcpp;
using namespace wavefield;

int main() {
    // --- 1. 定义波场和模拟参数 ---
    int Nx = 256, Ny = 256;
    float Lx_grid = 100.0f;
    float Ly_grid = 100.0f;
    float Hs = 0.5f;
    float Tp = 3.5f;
    float gamma = 3.3f;
    unsigned long seed = 1024;

    float t_start = 0.0f;
    float t_end = 100.0f;
    float dt = 0.1f;
    float t = t_start;

    float query_x = 0.0f;
    float query_y = 0.0f;

    std::cout << "--- 2D 波场单点持续查询绘图程序 ---\n";

    try {
        WaveFieldCalculator wave_calc(Nx, Ny, Lx_grid, Ly_grid, Hs, Tp, gamma, seed);

        std::vector<float> times;
        std::vector<float> wave_heights;
        std::vector<float> velocity_potentials;

        // 创建动态绘图窗口
        plt::ion();  // 开启交互模式
        plt::figure_size(800, 400);

        while (t <= t_end) {
            float eta = wave_calc.query_wave_height_point(query_x, query_y, t);
            float phi = wave_calc.query_velocity_potential_point(query_x, query_y, t);

            times.push_back(t);
            wave_heights.push_back(eta);
            velocity_potentials.push_back(phi);

            // 清空并重新绘制
            plt::clf();
            plt::subplot(2, 1, 1);
            plt::plot(times, wave_heights, "b-");
            plt::title("Wave Height vs Time");
            plt::xlabel("Time (s)");
            plt::ylabel("Wave Height (m)");

            plt::subplot(2, 1, 2);
            plt::plot(times, velocity_potentials, "r-");
            plt::title("Velocity Potential vs Time");
            plt::xlabel("Time (s)");
            plt::ylabel("Velocity Potential (m^2/s)");

            plt::pause(0.001);  // 暂停以刷新图形

            t += dt;

            // 可选：控制更新速度
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        plt::show();

    } catch (const std::exception& e) {
        std::cerr << "\n程序发生错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n程序发生未知错误。\n";
        return 1;
    }

    std::cout << "\n持续查询绘图测试结束。\n";
    return 0;
}
