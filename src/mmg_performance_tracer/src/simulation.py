#!/usr/bin/env python3
import subprocess
import time
import signal
import os
import sys
import re
import statistics

def launch_process(cmd, name, stdout=None):
    print(f"[INFO] Launching {name}...")
    return subprocess.Popen(
        cmd, stdout=stdout, stderr=subprocess.STDOUT, preexec_fn=os.setsid, text=False
    )

def terminate_process(proc, name):
    if proc and proc.poll() is None:
        print(f"[CLEANUP] Terminating {name}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            print(f"[CLEANUP] {name} terminated.")
        except Exception as e:
            print(f"[WARN] Failed to terminate {name}: {e}")

def run_simulation(batch_id, max_trials):
    optimization_times = []  # 保存每次 trial 的平均优化时间

    for trial in range(1, max_trials + 1):
        print(f"\n=== [Trial {trial}/{max_trials}] Starting Simulation Batch {batch_id} ===")

        sim_proc = controller_proc = tracer_proc = None
        try:
            # Step 1: 启动仿真器
            sim_proc = launch_process(["roslaunch", "mmg_simulator", "simulator.launch"], "Simulator")
            time.sleep(2)

            # Step 2: 启动控制器（带输出监听）
            controller_proc = launch_process(
                ["roslaunch", "mmg_mppi_controller", "simulation.launch"],
                "Controller + TrajGenerator",
                stdout=subprocess.PIPE  # 捕获输出
            )
            time.sleep(2)

            # Step 3: 启动跟踪器
            tracer_proc = launch_process(
                ["rosrun", "mmg_performance_tracer", "tracer.py"],
                "PerformanceTracer"
            )

            # Step 4: 实时读取控制器输出，记录优化日志行
            optimization_log_lines = []

            while True:
                raw_line = controller_proc.stdout.readline()
                if not raw_line:
                    break  # 子进程已结束，退出读取循环

                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")  # 替换非法字符

                print("[Controller]", line.strip())
                if "平均优化时间:" in line:
                    optimization_log_lines.append(line.strip())

            # 等待 controller 完全退出
            controller_proc.wait()
            print("[INFO] Controller + TrajGenerator exited.")

            # 提取最后一条优化日志
            avg_time_this_trial = None
            if optimization_log_lines:
                last_line = optimization_log_lines[-1]
                match = re.search(r"平均优化时间:\s*([\d.]+)\s*ms", last_line)
                if match:
                    avg_time_this_trial = float(match.group(1))
                    print(f"[解析] Trial {trial} 平均优化时间: {avg_time_this_trial} ms")
                    optimization_times.append(avg_time_this_trial) # 将优化时间添加到列表中
                else:
                    print("[WARN] 最后一条优化日志格式未匹配")
            else:
                print("[WARN] 未捕获到任何优化日志")

            # Step 5: 等待跟踪器退出
            tracer_proc.wait()
            print("[INFO] PerformanceTracer exited.")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Ctrl+C received. Cleaning up processes...")
            terminate_process(tracer_proc, "PerformanceTracer")
            terminate_process(controller_proc, "Controller + TrajGenerator")
            terminate_process(sim_proc, "Simulator")
            print("[EXIT] Aborted by user.")
            sys.exit(0)

        except Exception as e:
            print(f"[ERROR] Unexpected exception in trial {trial}: {e}")
            terminate_process(tracer_proc, "PerformanceTracer")
            terminate_process(controller_proc, "Controller + TrajGenerator")
            terminate_process(sim_proc, "Simulator")
            continue

        # Step 6: 正常终止仿真器
        terminate_process(sim_proc, "Simulator")
        print(f"[INFO] Trial {trial} finished.\n")
        time.sleep(2)

    # Step 7: 打印最终统计结果
    print("\n=== [RESULT] 优化时间统计 ===")
    if optimization_times:
        mean_time = statistics.mean(optimization_times)
        std_time = statistics.stdev(optimization_times) if len(optimization_times) > 1 else 0.0
        print(f"Trials: {len(optimization_times)}")
        print(f"平均优化时间: {mean_time:.2f} ms")
        print(f"标准差: {std_time:.2f} ms")
    else:
        print("未能提取任何优化时间数据。")

if __name__ == "__main__":
    try:
        N = 20  # 设置仿真次数
        run_simulation(batch_id="EXP01", max_trials=N)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Script manually interrupted. Exiting.")
        sys.exit(0)
