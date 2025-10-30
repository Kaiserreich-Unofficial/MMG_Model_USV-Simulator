#!/usr/bin/env python3
import subprocess
import time
import signal
import os
import sys
import re
import statistics
from tqdm import tqdm
import csv

def launch_process(cmd, name, cwd=None, stdout=None):
    print(f"[INFO] Launching {name} with command: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, text=False
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

def make_output_dir(param_dict):
    param_str = "_".join([f"{k}{str(v).replace(' ', '')}" for k, v in param_dict.items()])
    output_dir = os.path.join(os.getcwd(), param_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_simulation(max_trials, stddev_value):
    param_dict = {"StdDev": stddev_value}
    output_dir = make_output_dir(param_dict)
    print(f"[INFO] 输出目录: {output_dir}")

    optimization_times = []

    trial_iter = tqdm(range(1, max_trials + 1),
                      desc=f"StdDev{stddev_value}",
                      unit="trial")
    for trial in trial_iter:
        sim_proc = controller_proc = tracer_proc = None
        try:
            # 启动 Simulator
            sim_proc = launch_process(
                ["roslaunch", "mmg_simulator", "simulator-nogui.launch"],
                "Simulator",
                cwd=output_dir
            )
            time.sleep(1)

            # 启动 Controller，传递 stddev 参数
            controller_cmd = [
                "roslaunch", "mmg_mppi_controller", "simulation.launch",
                f"stddev:={stddev_value}"
            ]
            controller_proc = launch_process(
                controller_cmd, "Controller + TrajGenerator", cwd=output_dir, stdout=subprocess.PIPE)
            time.sleep(1)

            # 启动 PerformanceTracer
            tracer_proc = launch_process(
                ["rosrun", "mmg_performance_tracer", "tracer.py", output_dir],
                "PerformanceTracer",
                cwd=output_dir
            )

            # 解析优化时间日志
            last_opt_time_pattern = re.compile(
                r"平均优化时间: [\d.]+ ms, 上次优化时间: ([\d.]+) ms"
            )
            last_optimization_list = []

            while True:
                raw_line = controller_proc.stdout.readline()
                if not raw_line:
                    break
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")
                print("[Controller]", line.strip())

                match_last = last_opt_time_pattern.search(line)
                if match_last:
                    last_val = float(match_last.group(1))
                    last_optimization_list.append(last_val)
                    tqdm.write(f"[解析] Trial {trial} Step {len(last_optimization_list)} 上次优化时间: {last_val} ms")

            # 保存每 trial CSV
            if last_optimization_list:
                per_trial_csv = os.path.join(output_dir, f"last_opt_time_trial{trial}.csv")
                with open(per_trial_csv, mode="w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["StepIndex", "LastOptimizationTime_ms"])
                    for idx, val in enumerate(last_optimization_list, start=1):
                        writer.writerow([idx, val])
                tqdm.write(f"[INFO] 已保存 Trial {trial} 的优化时间到 {per_trial_csv}")

                # 收集平均优化时间
                optimization_times.append(statistics.mean(last_optimization_list))

            # 等待 Controller 正常退出
            try:
                controller_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tqdm.write("[WARN] Controller 超时，强制终止")
                terminate_process(controller_proc, "Controller + TrajGenerator (timeout)")

            # 等待 Tracer 正常退出
            try:
                tracer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tqdm.write("[WARN] Tracer 超时，强制终止")
                terminate_process(tracer_proc, "PerformanceTracer (timeout)")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Ctrl+C received. Cleaning up...")
            terminate_process(tracer_proc, "PerformanceTracer")
            terminate_process(controller_proc, "Controller + TrajGenerator")
            terminate_process(sim_proc, "Simulator")
            print("[EXIT] Aborted by user.")
            sys.exit(0)

        except Exception as e:
            tqdm.write(f"[ERROR] Trial {trial} 出现异常: {e}")
            terminate_process(tracer_proc, "PerformanceTracer")
            terminate_process(controller_proc, "Controller + TrajGenerator")
            terminate_process(sim_proc, "Simulator")
            continue

        terminate_process(sim_proc, "Simulator")
        time.sleep(2)

    # 统计整组 trial 优化时间
    tqdm.write(f"\n=== [RESULT] StdDev {stddev_value} 优化时间统计 ===")
    if optimization_times:
        mean_time = statistics.mean(optimization_times)
        std_time = statistics.stdev(optimization_times) if len(optimization_times) > 1 else 0.0
        result_str = f"""
        === [RESULT] StdDev {stddev_value} 优化时间统计 ===
        Trials: {len(optimization_times)}
        平均优化时间: {mean_time:.2f} ms
        标准差: {std_time:.2f} ms
        """
        tqdm.write(result_str.strip())
        result_file_path = os.path.join(output_dir, "optimization_stats.txt")
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(result_str)
    else:
        tqdm.write("未能提取任何优化时间数据。")

def main():
    stddev_list = [1.0, 5.0, 10.0, 15.0]
    max_trials = 30
    print(f"[INFO] 总共要跑 {len(stddev_list)} 组 stddev，每组 {max_trials} 次仿真")

    try:
        for stddev_value in stddev_list:
            run_simulation(max_trials, stddev_value)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 脚本被手动终止，退出。")
        sys.exit(0)

if __name__ == "__main__":
    main()
