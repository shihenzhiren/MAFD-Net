import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import subprocess
import re
import uuid
import logging
import socket
import time
import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO)
logging.getLogger('hpbandster').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class MyWorker(Worker):
    def __init__(self, *args, fixed_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_param = fixed_param  # 固定参数，例如 {'beta_crd': 0.0}
        logger.debug("Worker 初始化完成")

    def compute(self, config, budget, **kwargs):
        # 获取超参数，合并固定参数
        beta_kd   = config.get('beta_kd',   self.fixed_param.get('beta_kd',   0.0))
        beta_hint = config.get('beta_hint',  self.fixed_param.get('beta_hint', 0.0))
        beta_crd  = config.get('beta_crd',   self.fixed_param.get('beta_crd',  0.0))
        epochs = int(budget)
        trial_id = config.get('trial_id', f"bohb_{str(uuid.uuid4())[:8]}")

        logger.info(f"开始试验 {trial_id}，参数：beta_kd={beta_kd}, beta_hint={beta_hint}, beta_crd={beta_crd}, epochs={epochs}")

        cmd = [
            "python", "train_student.py",
            "--path_t", "./save/teachers/models/EfficientNetB7_vanilla_QAX2024_trial_B7/EfficientNetB7_best.pth",
            "--distill", "kd", "hint", "crd",
            "--model_s", "EfficientNetB0_t",
            "-c", "1",
            "-d", "1",
            "-b", "0.8",
            "--trial", trial_id,
            "--gpu_id", "0",
            "--dataset", "QAX2024",
            "--beta_kd",   str(beta_kd),
            "--beta_hint", str(beta_hint),
            "--beta_crd",  str(beta_crd),
            "--hint_layer", "2",
            "--temperature", "0.1",
            "--weight_schedule", "linear",
            "--weight_schedule_start", "0.0",
            "--weight_schedule_end", "0.8",
            "--nce_k", "4096",
            "--epochs", str(epochs),
            "--batch_size", "64"
        ]

        try:
            logger.debug(f"执行命令: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()
            )

            output = []
            timeout_seconds = 7200
            start_time = time.time()

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end='', flush=True)
                    output.append(line)
                if time.time() - start_time > timeout_seconds:
                    process.terminate()
                    raise TimeoutError(f"试验 {trial_id} 超时，超过 {timeout_seconds} 秒")

            returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd, ''.join(output))

            output_str = ''.join(output)

            # 在 compute 方法中
            metrics_file = os.path.join(
                f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_QAX2024_kd+hint+crd_r:1.0_a:1.0_b:0.8_{trial_id}",
                "test_best_metrics.json"
            )
            logger.debug(f"检查 JSON 文件路径: {metrics_file}")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    logger.debug(f"JSON 文件内容: {metrics}")
                    test_acc = float(metrics.get('test_acc', 0.0))
                    logger.info(f"试验 {trial_id} 完成，测试准确率={test_acc} (从 JSON 文件解析，最优结果)")
            else:
                # 回退到终端输出解析
                match = re.findall(r'\*\*?\s*Acc@1\s*[=:]?\s*([0-9]+\.[0-9]+)', output_str)
                if match:
                    test_acc = max(float(val) for val in match)
                    logger.info(f"试验 {trial_id} 完成，测试准确率={test_acc} (从终端输出解析，最高值)")
                else:
                    test_acc = 0.0
                    logger.warning(f"试验 {trial_id} 无法解析准确率，默认值为 0.0")

            if test_acc >= 100.0 and test_acc != metrics.get('test_acc_top5', 0.0):
                logger.warning(f"试验 {trial_id} 的 test_acc={test_acc} 不合理，设为 0.0")
                test_acc = 0.0
            '''#match = re.search(r'Acc@1\s+([0-9]+\.[0-9]+)', output_str)
            match = re.search(r'\*\*?\s*Acc@1\s*[=:]?\s*([0-9]+\.[0-9]+)', output_str.split('\n')[-1])
            if match:
                test_acc = float(match.group(1))
                logger.info(f"试验 {trial_id} 完成，测试准确率={test_acc} (从终端输出解析)")
            else:
                metrics_file = os.path.join(
                    f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_nsl_kd+afd+crd_r:1.0_a:1.0_b:0.8_{trial_id}",
                    "test_best_metrics.json"
                )
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        test_acc = float(metrics.get('test_acc', 0.0))
                        logger.info(f"试验 {trial_id} 完成，测试准确率={test_acc} (从JSON文件解析)")
                else:
                    test_acc = 0.0
                    logger.warning(f"试验 {trial_id} 无法解析测试准确率，默认值为0.0")'''

        except (subprocess.CalledProcessError, TimeoutError) as e:
            logger.error(f"试验 {trial_id} 失败，错误：{e}")
            test_acc = 0.0
            output_str = getattr(e, 'output', '')

        return {
            'loss': -test_acc,
            'info': {'test_acc': test_acc, 'output': output_str, 'trial_id': trial_id}
        }

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_teacher_name(model_path):
    directory = model_path.split(os.sep)[-2]
    # 修正正则表达式
    pattern = r'S:(.+?)_T:'
    name_match = re.search(pattern, directory)
    if name_match:
        return name_match.group(1)
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return '_'.join(segments[:3])
    return segments[0]

def plot_heatmap(results, param1, param2, fixed_param, output_dir):
    """绘制两两超参数的热力图"""
    os.makedirs(output_dir, exist_ok=True)
    param1_values = sorted(set(r[param1] for r in results))
    param2_values = sorted(set(r[param2] for r in results))
    heatmap_data = np.zeros((len(param1_values), len(param2_values)))

    for r in results:
        i = param1_values.index(r[param1])
        j = param2_values.index(r[param2])
        heatmap_data[i, j] = r['test_acc']

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, xticklabels=[f"{v:.2f}" for v in param2_values], yticklabels=[f"{v:.2f}" for v in param1_values], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f"Test Accuracy Heatmap ({fixed_param} fixed)")
    plt.savefig(os.path.join(output_dir, f"heatmap_{param1}_{param2}.png"))
    plt.close()

def run_bohb_combination(mode, fixed_param, param_names, result_dir, min_budget=5, max_budget=10, n_iterations=20):
    """运行 BOHB 优化"""
    cs = CS.ConfigurationSpace()
    for param in param_names:
        if param == 'beta_kd':
            cs.add(CSH.UniformFloatHyperparameter('beta_kd', lower=0.1, upper=1.0, default_value=0.5))
        elif param == 'beta_hint':
            cs.add(CSH.UniformFloatHyperparameter('beta_hint', lower=0.1, upper=5.0, default_value=1.0))
        elif param == 'beta_crd':
            cs.add(CSH.UniformFloatHyperparameter('beta_crd', lower=0.1, upper=5.0, default_value=1.0))

    port = find_free_port()
    logger.info(f"模式 {mode}，优化参数 {param_names}，固定参数 {fixed_param}，使用端口: {port}")

    ns = hpns.NameServer(run_id='bohb_search', host='127.0.0.1', port=port)
    try:
        ns.start()
        logger.debug("NameServer 启动成功")
    except Exception as e:
        logger.error(f"NameServer启动失败: {e}")
        return None

    worker = MyWorker(nameserver='127.0.0.1', nameserver_port=port, run_id='bohb_search', fixed_param=fixed_param)
    try:
        worker_thread = worker.run(background=True)
        logger.debug("Worker 启动成功")
    except Exception as e:
        logger.error(f"Worker启动失败: {e}")
        ns.shutdown()
        return None

    bohb = BOHB(
        configspace=cs,
        run_id='bohb_search',
        nameserver='127.0.0.1',
        nameserver_port=port,
        min_budget=min_budget,
        max_budget=max_budget,
        num_samples=64,
        random_fraction=0.33
    )

    try:
        result = bohb.run(n_iterations=n_iterations)
        logger.info(f"模式 {mode} 优化完成")

        id2config = result.get_id2config_mapping()
        incumbent = result.get_incumbent_id()
        if incumbent is None:
            logger.error("未找到最优配置（incumbent），优化失败。")
            return None
        # 取最优 run（loss 最小）
        runs = result.get_runs_by_id(incumbent)
        best_run = min(runs, key=lambda r: r.loss)
        best_config = id2config[incumbent]['config']
        best_acc = -best_run.loss
        best_trial_id = best_run.info.get('trial_id', 'unknown')

        # 保存结果
        os.makedirs(result_dir, exist_ok=True)
        # 修正 all_runs 收集方式
        all_runs = []
        # 在函数开头定义全局最优
        global_best_acc = -float('inf')
        global_best_trial_dir = None

        for run in result.get_all_runs():
            config = id2config[run.config_id]['config']
            test_acc = -run.loss
            trial_id = run.info.get('trial_id', 'unknown')
            trial_dir = f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_QAX2024_kd+hint+crd_r:1.0_a:1.0_b:0.8_{trial_id}"
            all_runs.append({
                "config": config,
                "test_acc": test_acc,
                "trial_id": trial_id
            })
            if test_acc > global_best_acc:
                global_best_acc = test_acc
                global_best_trial_dir = trial_dir
                # 复制权重和metrics到bohb_results/best_model
                import shutil
                best_model_dir = os.path.join(result_dir, "best_model")
                if os.path.exists(best_model_dir):
                    shutil.rmtree(best_model_dir)
                shutil.copytree(trial_dir, best_model_dir)
                logger.info(f"已更新全局最优模型到 {best_model_dir}，Acc@1={test_acc}")

        combination_result = {
            "mode": mode,
            "fixed_param": fixed_param,
            "best_config": best_config,
            "best_acc": best_acc,
            "best_trial_id": best_trial_id,
            "model_path": f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_QAX2024_kd+hint+crd_r:1.0_a:1.0_b:0.8_{best_trial_id}/EfficientNetB0_t_best.pth",
            "metrics_path": f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_QAX2024_kd+hint+crd_r:1.0_a:1.0_b:0.8_{best_trial_id}/test_best_metrics.json",
            "all_runs": all_runs
        }
        result_file = os.path.join(result_dir, f"result_{mode}.json")
        with open(result_file, "w") as f:
            json.dump(combination_result, f, indent=4)
        logger.info(f"模式 {mode} 结果已保存到 {result_file}")

        # 如果是两两组合，绘制热力图
        if mode != "all":
            # fixed_param 只显示 key
            fixed_param_str = ', '.join([f"{k}={v}" for k, v in fixed_param.items()])
            plot_heatmap(combination_result["all_runs"], param_names[0], param_names[1], fixed_param_str, result_dir)

        # 清理次优模型
        for run_id in id2config:
            if run_id != incumbent:
                trial_id = id2config[run_id]['config'].get('trial_id', None)
                if trial_id:
                    trial_dir = f"./save/students/models/S:EfficientNetB0_t_T:EfficientNetB7_QAX2024_kd+hint+crd_r:1.0_a:1.0_b:0.8_{trial_id}"
                    if os.path.exists(trial_dir):
                        shutil.rmtree(trial_dir)
                        logger.info(f"删除次优试验目录: {trial_dir}")

        return combination_result

    except Exception as e:
        import traceback
        logger.error(f"模式 {mode} 优化失败: {e}\n{traceback.format_exc()}")
        return None
    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run BOHB optimization for hyperparameter combinations")
    parser.add_argument("--mode", type=str, choices=["kd_hint", "kd_crd", "hint_crd", "all"], default="all",
                        help="Optimization mode: kd_hint, kd_crd, hint_crd, or all")
    args = parser.parse_args()

    result_dir = "./save/bohb_results"

    # 定义组合
    combinations = {
        "kd_hint":  (["beta_kd", "beta_hint"], {"beta_crd":  0.0}),
        "kd_crd":   (["beta_kd", "beta_crd"],  {"beta_hint": 0.0}),
        "hint_crd": (["beta_hint", "beta_crd"], {"beta_kd":   0.0}),
        "all":      (["beta_kd", "beta_hint", "beta_crd"], {})
    }

    # 根据 mode 选择组合
    param_names, fixed_param = combinations[args.mode]
    logger.info(f"运行模式: {args.mode}, 优化参数: {param_names}, 固定参数: {fixed_param}")

    # 运行优化
    result = run_bohb_combination(args.mode, fixed_param, param_names, result_dir, min_budget=5, max_budget=10, n_iterations=20)

    # 保存结果汇总
    if result:
        os.makedirs(result_dir, exist_ok=True)
        summary = {
            "mode": args.mode,
            "fixed_param": fixed_param,
            "best_config": result["best_config"],
            "best_acc": result["best_acc"],
            "best_trial_id": result["best_trial_id"],
            "model_path": result["model_path"],
            "metrics_path": result["metrics_path"]
        }
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"结果汇总已保存到 {os.path.join(result_dir, 'summary.json')}")