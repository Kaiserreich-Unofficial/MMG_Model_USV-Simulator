#include "extwave.hpp"
#include <cmath>
#include <stdexcept>

namespace waveforce
{

    // ------------------------------------------------------------
    // 构造与采样布局
    // ------------------------------------------------------------
    WaveForceCalculator::WaveForceCalculator(wavefield::WaveFieldCalculator &wf,
                                             float L, float B, float draft,
                                             float rho,
                                             int n_span, int n_vert)
        : wf_(wf), L_(L), B_(B), draft_(draft), rho_(rho),
          n_span_(std::max(1, n_span)), n_vert_(std::max(1, n_vert)),
          last_t_(0.0f), have_last_(false)
    {
        build_sample_grid();
    }

    void WaveForceCalculator::set_sampling(int n_span, int n_vert)
    {
        n_span_ = std::max(1, n_span);
        n_vert_ = std::max(1, n_vert);
        build_sample_grid();
    }

    void WaveForceCalculator::build_sample_grid()
    {
        // 清理历史
        have_last_ = false;
        last_t_ = 0.0f;
        face_samples_.clear();
        last_phi_surface_.clear();

        // 四个垂直面： +X (bow), -X (stern), +Y (starboard), -Y (port)
        // 在艇本体坐标系中定义面位置：
        // 面在 x 方向的截面： x = +L/2 或 x = -L/2，跨越 y ∈ [-B/2, B/2], z ∈ [-draft, 0]
        // 面在 y 方向的截面： y = +B/2 或 y = -B/2，跨越 x ∈ [-L/2, L/2], z ∈ [-draft, 0]
        // 对每个面生成 n_span_ × n_vert_ 个采样点 (均匀)
        // 存储 (local_coord1, z) —— 本实现统一为 local span coordinate (沿面展开) 与 z

        // face order: +X, -X, +Y, -Y
        for (int face = 0; face < 4; ++face)
        {
            for (int is = 0; is < n_span_; ++is)
            {
                float frac_s = (n_span_ == 1) ? 0.5f : (float)is / (float)(n_span_ - 1); // 0..1
                for (int iv = 0; iv < n_vert_; ++iv)
                {
                    float frac_z = (n_vert_ == 1) ? 0.5f : (float)iv / (float)(n_vert_ - 1); // 0..1
                    float z = -draft_ * (1.0f - frac_z);                                     // z from -draft .. 0, but prefer -draft..0 mapped as frac_z
                    // we'll store two coords: span-position (to be interpreted per face) and z
                    face_samples_.push_back({frac_s, z});
                    last_phi_surface_.push_back(0.0f); // 初始化历史势
                }
            }
        }
        // last_phi_surface_ 长度 = 4 * n_span_ * n_vert_
    }

    // ------------------------------------------------------------
    // 体坐标 -> 世界坐标变换
    // local_x, local_y 在艇本体坐标系（艇轴向为 +x_body，右舷为 +y_body）
    // ------------------------------------------------------------
    void WaveForceCalculator::world_from_body(float bx, float by, float psi,
                                              float local_x, float local_y,
                                              float &wx, float &wy) const
    {
        float c = cosf(psi);
        float s = sinf(psi);
        wx = bx + local_x * c - local_y * s;
        wy = by + local_x * s + local_y * c;
    }

    // ------------------------------------------------------------
    // 将世界坐标映射到 wavefield 的栅格索引（按你 wavefield 的网格方式）
    // 使用 wavefield 的 Lx, Ly, Nx, Ny 信息
    // ------------------------------------------------------------
    size_t WaveForceCalculator::grid_index_from_world(float wx, float wy) const
    {
        float Lx = wf_.get_Lx();
        float Ly = wf_.get_Ly();
        int Nx = wf_.get_Nx();
        int Ny = wf_.get_Ny();

        float dx = Lx / (float)Nx;
        float dy = Ly / (float)Ny;

        // 对世界坐标做周期映射到 [0, L)
        float x = fmodf(wx, Lx);
        float y = fmodf(wy, Ly);
        if (x < 0.0f)
            x += Lx;
        if (y < 0.0f)
            y += Ly;

        int m = static_cast<int>(roundf(x / dx));
        int n = static_cast<int>(roundf(y / dy));
        if (m == Nx)
            m = 0;
        if (n == Ny)
            n = 0;
        return (size_t)n * (size_t)Nx + (size_t)m;
    }

    // ------------------------------------------------------------
    // 主要计算函数
    // ------------------------------------------------------------
    ForceResult WaveForceCalculator::compute_force(float t, float x, float y, float psi)
    {
        ForceResult out;
        // 预计算
        const int Nx = wf_.get_Nx();
        const int Ny = wf_.get_Ny();
        const float Lx = wf_.get_Lx();
        const float Ly = wf_.get_Ly();

        // per-sample area on each face:
        // 对于四个面，面积都相等： A_face = span_length * draft
        // span_length depends on face orientation: for +X/-X faces span = B, for +Y/-Y faces span = L
        float dA_xface = (B_ / (float)std::max(1, n_span_)) * (draft_ / (float)std::max(1, n_vert_));
        float dA_yface = (L_ / (float)std::max(1, n_span_)) * (draft_ / (float)std::max(1, n_vert_));

        // iterate faces and their samples
        int samples_per_face = n_span_ * n_vert_;
        // indices through face_samples_ and last_phi_surface_
        size_t idx_global = 0;
        for (int face = 0; face < 4; ++face)
        {
            // face local geometry
            bool is_x_face = (face == 0 || face == 1);
            float face_x = 0.0f, face_y = 0.0f;
            float span_len = is_x_face ? B_ : L_;
            float half_L = L_ * 0.5f;
            float half_B = B_ * 0.5f;

            // face center offset in body coords
            if (face == 0) // +X
                face_x = +half_L;
            else if (face == 1) // -X
                face_x = -half_L;
            else if (face == 2) // +Y
                face_y = +half_B;
            else // face == 3 // -Y
                face_y = -half_B;

            // normal in body coords
            float nx_body = 0.0f, ny_body = 0.0f;
            if (face == 0)
            {
                nx_body = 1.0f;
                ny_body = 0.0f;
            }
            if (face == 1)
            {
                nx_body = -1.0f;
                ny_body = 0.0f;
            }
            if (face == 2)
            {
                nx_body = 0.0f;
                ny_body = 1.0f;
            }
            if (face == 3)
            {
                nx_body = 0.0f;
                ny_body = -1.0f;
            }

            // convert normal to world coords by rotation
            float cpsi = cosf(psi), spsi = sinf(psi);
            float nx_world = nx_body * cpsi - ny_body * spsi;
            float ny_world = nx_body * spsi + ny_body * cpsi;

            for (int s = 0; s < samples_per_face; ++s, ++idx_global)
            {
                // sample info stored as (frac_along_span, z)
                float frac_s = face_samples_[idx_global][0];
                float z = face_samples_[idx_global][1]; // negative downwards

                // compute local (body) coordinates for the sample point
                float local_x = 0.0f, local_y = 0.0f;
                if (is_x_face)
                {
                    local_x = face_x;
                    // span along y from -B/2 .. +B/2
                    local_y = -half_B + frac_s * span_len;
                }
                else
                {
                    local_y = face_y;
                    // span along x from -L/2 .. +L/2
                    local_x = -half_L + frac_s * span_len;
                }

                // transform to world coordinates
                float wx, wy;
                world_from_body(x, y, psi, local_x, local_y, wx, wy);

                // obtain surface φ at this horizontal location and time t
                float phi_surf = 0.0f;
                try
                {
                    phi_surf = wf_.query_velocity_potential_point(wx, wy, t);
                }
                catch (const std::exception &e)
                {
                    // propagate with context
                    throw std::runtime_error(std::string("wavefield query failed: ") + e.what());
                }

                // compute dphi/dt by backward difference if available
                float dphi_dt_surf = 0.0f;
                if (have_last_)
                {
                    float dt = t - last_t_;
                    if (dt <= 0.0f)
                        dt = 1e-6f;
                    dphi_dt_surf = (phi_surf - last_phi_surface_[idx_global]) / dt;
                }
                else
                {
                    dphi_dt_surf = 0.0f; // first call -> cannot compute derivative
                }

                // retrieve local wave number k from wavefield grid (approx)
                // map world coord to grid index and then to h_k_abs
                size_t grid_idx = grid_index_from_world(wx, wy);
                const auto &h_k = wf_.get_h_k_abs();
                // grid_idx might be out of range if wavefield's stored vectors differ in flattening,
                // but per construction wavefield uses Nx * Ny flattening -> OK
                float k_local = 0.0f;
                if (grid_idx < h_k.size())
                    k_local = h_k[grid_idx];
                // ensure non-negative
                if (k_local < 1e-6f)
                    k_local = 1e-6f;

                // vertical decay factor exp(k * z) where z is negative
                float decay = expf(k_local * z); // z <= 0, so decay <= 1

                // pressure at depth z: p = - rho * (∂Φ/∂t)_z = - rho * dphi_dt_surf * decay
                float p = -rho_ * dphi_dt_surf * decay;

                // differential area
                float dA = is_x_face ? dA_xface : dA_yface;

                // contribution to force (world coords)
                float dFx = p * nx_world * dA;
                float dFy = p * ny_world * dA;

                out.Fx += dFx;
                out.Fy += dFy;

                // moment about center (z-axis): r × F where r = (rx, ry, 0), F = (dFx, dFy, 0)
                // rx, ry are vector from body center to sample point in world coords
                float rx_world = (wx - x);
                float ry_world = (wy - y);
                float dMz = rx_world * dFy - ry_world * dFx;
                out.Mz += dMz;

                // store phi for next derivative
                last_phi_surface_[idx_global] = phi_surf;
            }
        }

        // update last_t_ flag
        last_t_ = t;
        have_last_ = true;

        return out;
    }

} // namespace waveforce
