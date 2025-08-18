#ifndef USV_DYNAMIC_CUH
#define USV_DYNAMIC_CUH

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>
#include "FxTDO.cuh"

namespace heron
{
    // 水动力参数结构体
    struct HydroDynamicParams
    {
        float mass; // 质量
        float Iz;   // 转动惯量
        float B;    // 桨距
        float X_u_dot;
        float Y_v_dot;
        float N_r_dot;
        float X_u;
        float Y_v;
        float N_r;
    };

    struct USVDynamicsParams : public DynamicsParams
    {
        enum class StateIndex : int
        {
            POS_X = 0,
            POS_Y,
            POS_PSI,
            VEL_U,
            VEL_V,
            VEL_R,
            NUM_STATES = 6
        };
        enum class ControlIndex : int
        {
            INPUT_LEFT = 0, // S_left
            INPUT_RIGHT,    // S_right
            NUM_CONTROLS = 2
        };
        enum class OutputIndex : int
        {
            POS_X = 0,
            POS_Y,
            POS_PSI,
            VEL_U,
            VEL_V,
            VEL_R,
            NUM_OUTPUTS = 6
        };
        USVDynamicsParams() = default;
        ~USVDynamicsParams() = default;
    };

    using namespace MPPI_internal;

    class USVDynamics : public Dynamics<USVDynamics, USVDynamicsParams>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        USVDynamics(cudaStream_t stream = nullptr);

        void setDynamicsParams(const HydroDynamicParams &hydroparams, const float &input_limit, const float &substep);

        std::string getDynamicsModelName() const override
        {
            return "SmallYellowBoat Dynamics";
        }

        void computeDynamics(const Eigen::Ref<const state_array> &state, const Eigen::Ref<const control_array> &control,
                             Eigen::Ref<state_array> state_der);

        void printState(float *state);
        void printState(const float *state);

        __device__ void computeDynamics(float *state, float *control, float *state_der, float *theta = nullptr);

        void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
                  const Eigen::Ref<const control_array> &control, Eigen::Ref<output_array> output, const float t,
                  const float dt);

        __device__ inline void step(float *state, float *next_state, float *state_der, float *control, float *output, float *theta_s, const float t, const float dt);

        state_array stateFromMap(const std::map<std::string, float> &map);

        // 施加控制约束
        __host__ void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control);
        __device__ void enforceConstraints(float *state, float *control);
        bool enable_fxtdo_;      // 是否启用扰动状态观测器
        FxTDO fxtdo_; // 扰动状态观测器
        float fxtdo_alpha_; // 可调节，控制低通滤波带宽
    private:
        // 自定义参数
        HydroDynamicParams hydroparams_; // 水动力参数
        float input_limit_;              // 输入限制
        float substep_;                  // 子步长
        Eigen::Matrix3f inv_M_;          // 惯量矩阵
        Eigen::Matrix3f D_;              // 阻尼矩阵
        float d_hat_filtered_prev[3] = {0.0f, 0.0f, 0.0f}; // 上一时刻的滤波后的扰动速度
    };
}

#if __CUDACC__
#include "usv_dynamics.cu"
#endif

#endif // USV_DYNAMIC_CUH
