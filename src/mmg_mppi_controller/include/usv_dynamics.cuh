#ifndef USV_DYNAMIC_CUH
#define USV_DYNAMIC_CUH

#include <mppi/utils/math_utils.h>
#include <mppi/dynamics/dynamics.cuh>
#include "FxTDO.cuh"

namespace heron
{
    struct HydroDynamicParams
    {
        float mass;
        float Iz;
        float B;
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
            POS_X = 0, POS_Y, POS_PSI, VEL_U, VEL_V, VEL_R, NUM_STATES = 6
        };
        enum class ControlIndex : int
        {
            INPUT_LEFT = 0, INPUT_RIGHT, NUM_CONTROLS = 2
        };
        enum class OutputIndex : int
        {
            POS_X = 0, POS_Y, POS_PSI, VEL_U, VEL_V, VEL_R, NUM_OUTPUTS = 6
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

        std::string getDynamicsModelName() const override;

        // --------------- 新增方法 ----------------
        // 计算 tau_eff
        __host__ __device__ void computeTauEff(const float *state, float Tl, float Tr, float *tau_eff) const;

        // --------------- 动力学 ----------------
        void computeDynamics(const Eigen::Ref<const state_array> &state,
                             const Eigen::Ref<const control_array> &control,
                             Eigen::Ref<state_array> state_der,
                             FxTDOState &fxtdo_state);

        __device__ void computeDynamics(float *state, float *control, float *state_der,
                                        FxTDOState &fxtdo_state, float *theta = nullptr);

        void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                  Eigen::Ref<state_array> state_der,
                  const Eigen::Ref<const control_array> &control, Eigen::Ref<output_array> output,
                  const float t, const float dt);

        __device__ inline void step(float *state, float *next_state, float *state_der,
                                    float *control, float *output, float *theta_s,
                                    const float t, const float dt);

        state_array stateFromMap(const std::map<std::string, float> &map);

        void printState(float *state);
        void printState(const float *state);

        // 控制约束
        __host__ void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control);
        __device__ void enforceConstraints(float *state, float *control);

    public:
        bool enable_fxtdo_;
        FxTDO fxtdo_;
        FxTDOState shared_fxtdo_state;

    private:
        HydroDynamicParams hydroparams_;
        float input_limit_;
        float substep_;
        Eigen::Matrix3f inv_M_;
        Eigen::Matrix3f D_;
    };
}

#if __CUDACC__
#include "usv_dynamics.cu"
#endif

#endif // USV_DYNAMIC_CUH
