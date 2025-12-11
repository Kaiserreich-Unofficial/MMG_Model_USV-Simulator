#pragma once

#include "wavefield.cuh" // 你的 wavefield 头（前文提供的那个）
#include <vector>
#include <array>

namespace waveforce
{

struct ForceResult
{
    float Fx = 0.0f;   // 世界坐标系 X 力 (N)
    float Fy = 0.0f;   // 世界坐标系 Y 力 (N)
    float Mz = 0.0f;   // 围绕艇质心的垂直力矩 (N·m), 正值按右手定则
};

class WaveForceCalculator
{
public:
    // 构造：
    // wf           : 已初始化的 wavefield::WaveFieldCalculator 的引用（必须在其生命周期内有效）
    // L (m)        : 船体在船首向的长度（x_body）
    // B (m)        : 船体横向宽度（y_body）
    // draft (m)    : 吃水（浸没深度，正数）
    // rho (kg/m^3) : 密度，默认淡水 1000
    // n_span, n_vert: 每个垂直面沿横向和竖向的数值积分采样数（越大越精确，计算越慢）
    WaveForceCalculator(wavefield::WaveFieldCalculator &wf,
                        float L, float B, float draft,
                        float rho = 1000.0f,
                        int n_span = 6, int n_vert = 6);

    // 计算 Froude–Krylov 力
    // t, x, y, psi : 当前时刻与艇在世界坐标系下的位置与航向 (psi: rad)
    // 返回 ForceResult (Fx,Fy,Mz)
    // 注意：第一次调用会用上一次没有历史帧的后向差分 -> 返回力接近 0（并启动历史状态）。之后的调用将返回可用的值。
    ForceResult compute_force(float t, float x, float y, float psi);

    // 调整采样密度（运行时安全）
    void set_sampling(int n_span, int n_vert);

private:
    wavefield::WaveFieldCalculator &wf_;
    float L_, B_, draft_, rho_;
    int n_span_, n_vert_;

    // 记录用于有限差分的上一帧时间与势值（按采样点顺序）
    float last_t_;
    bool have_last_;
    std::vector<float> last_phi_surface_; // φ at surface at each sample point

    // 采样点在艇本体坐标系下的位置（y and z only for each face sample）
    // we precompute local sample offsets for 4 vertical faces
    struct SamplePoint { float sx_world; float sy_world; float sz; }; // sz negative downward from free surface
    // For each face: vector of (local_y, local_z) for samples; x is face offset (±L/2 or ±B/2)
    std::vector<std::array<float,2>> face_samples_; // length = 4 * (n_span_ * n_vert_) flattened
    void build_sample_grid(); // rebuilds sampling layout and resets history

    // helpers
    void world_from_body(float bx, float by, float psi, float local_x, float local_y, float &wx, float &wy) const;
    size_t grid_index_from_world(float wx, float wy) const;
};
} // namespace waveforce
