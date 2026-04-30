# results_visualization.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy import stats
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class BOHBResultsVisualizer:
    def __init__(self, result_dir="./save/bohb_results"):
        self.result_dir = result_dir
        self.results = {}
        self.summary_data = []
        
    def load_results(self):
        """加载所有结果文件"""
        if not os.path.exists(self.result_dir):
            raise FileNotFoundError(f"结果目录不存在: {self.result_dir}")
        
        # 加载汇总文件
        summary_file = os.path.join(self.result_dir, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
        
        # 加载各个模式的结果
        modes = ["kd_afd", "kd_crd", "afd_crd", "all"]
        for mode in modes:
            result_file = os.path.join(self.result_dir, f"result_{mode}.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    self.results[mode] = json.load(f)
                    self.summary_data.append({
                        'mode': mode,
                        'best_acc': self.results[mode]['best_acc'],
                        'config': self.results[mode]['best_config']
                    })
    
    def plot_performance_comparison(self, output_path=None):
        """绘制不同组合的性能对比图"""
        if not self.summary_data:
            print("没有可用的汇总数据")
            return
        
        df = pd.DataFrame(self.summary_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(df)), df['best_acc'], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
        
        ax.set_xlabel('蒸馏方法组合', fontsize=14)
        ax.set_ylabel('测试准确率 (%)', fontsize=14)
        ax.set_title('不同知识蒸馏组合的性能对比', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([self._get_mode_label(mode) for mode in df['mode']], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {output_path}")
        plt.show()
    
    def _get_mode_label(self, mode):
        """获取模式标签"""
        labels = {
            'kd_afd': 'KD + AFD',
            'kd_crd': 'KD + CRD',
            'afd_crd': 'AFD + CRD',
            'all': 'KD + AFD + CRD'
        }
        return labels.get(mode, mode)
    
    def plot_hyperparameter_heatmaps(self, output_dir=None):
        """绘制超参数热力图"""
        if output_dir is None:
            output_dir = os.path.join(self.result_dir, "heatmaps")
        os.makedirs(output_dir, exist_ok=True)
        
        for mode, result in self.results.items():
            if mode != "all" and 'all_runs' in result:
                runs = result['all_runs']
                if runs:
                    # 提取参数和准确率
                    param1 = list(result['best_config'].keys())[0]
                    param2 = list(result['best_config'].keys())[1]
                    
                    param1_vals = sorted(set(run['config'][param1] for run in runs))
                    param2_vals = sorted(set(run['config'][param2] for run in runs))
                    
                    # 创建热力图数据
                    heatmap_data = np.zeros((len(param1_vals), len(param2_vals)))
                    
                    for run in runs:
                        i = param1_vals.index(run['config'][param1])
                        j = param2_vals.index(run['config'][param2])
                        heatmap_data[i, j] = run['test_acc']
                    
                    # 绘制热力图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto')
                    
                    # 设置坐标轴
                    ax.set_xticks(range(len(param2_vals)))
                    ax.set_yticks(range(len(param1_vals)))
                    ax.set_xticklabels([f'{x:.2f}' for x in param2_vals])
                    ax.set_yticklabels([f'{x:.2f}' for x in param1_vals])
                    
                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('测试准确率 (%)', rotation=270, labelpad=20)
                    
                    ax.set_xlabel(param2.replace('beta_', '').upper())
                    ax.set_ylabel(param1.replace('beta_', '').upper())
                    ax.set_title(f'{self._get_mode_label(mode)} 超参数热力图', fontsize=14)
                    
                    # 添加数值标注
                    for i in range(len(param1_vals)):
                        for j in range(len(param2_vals)):
                            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                                          ha="center", va="center", 
                                          color="white" if heatmap_data[i, j] > np.mean(heatmap_data) else "black")
                    
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f"heatmap_{mode}.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"热力图已保存到: {output_path}")
    
    def plot_convergence_curve(self, output_path=None):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mode, result in self.results.items():
            if 'all_runs' in result:
                runs = result['all_runs']
                # 按试验ID排序（假设试验ID包含时间信息）
                runs_sorted = sorted(runs, key=lambda x: x['trial_id'])
                accuracies = [run['test_acc'] for run in runs_sorted]
                
                # 计算累积最大值
                cumulative_max = np.maximum.accumulate(accuracies)
                
                ax.plot(range(1, len(cumulative_max) + 1), cumulative_max, 
                       label=self._get_mode_label(mode), linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('试验次数', fontsize=14)
        ax.set_ylabel('最佳测试准确率 (%)', fontsize=14)
        ax.set_title('BOHB优化收敛曲线', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存到: {output_path}")
        plt.show()
    
    def plot_hyperparameter_distribution(self, output_path=None):
        """绘制超参数分布图"""
        all_runs = []
        for mode, result in self.results.items():
            if 'all_runs' in result:
                for run in result['all_runs']:
                    run_data = run['config'].copy()
                    run_data['test_acc'] = run['test_acc']
                    run_data['mode'] = mode
                    all_runs.append(run_data)
        
        if not all_runs:
            return
        
        df = pd.DataFrame(all_runs)
        
        # 绘制箱线图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        params = ['beta_kd', 'beta_afd', 'beta_crd']
        titles = ['KD权重分布', 'AFD权重分布', 'CRD权重分布']
        
        for i, (param, title) in enumerate(zip(params, titles)):
            if param in df.columns:
                # 过滤掉0值（固定参数）
                data = df[df[param] > 0]
                if not data.empty:
                    sns.boxplot(x='mode', y=param, data=data, ax=axes[i])
                    axes[i].set_title(title, fontsize=14)
                    axes[i].set_xlabel('')
                    axes[i].set_xticklabels([self._get_mode_label(mode) for mode in data['mode'].unique()], 
                                          rotation=45, ha='right')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"超参数分布图已保存到: {output_path}")
        plt.show()
    
    def create_results_table(self, output_path=None):
        """创建结果表格"""
        table_data = []
        
        for mode, result in self.results.items():
            if 'best_config' in result:
                row = {
                    '方法组合': self._get_mode_label(mode),
                    '测试准确率 (%)': f"{result['best_acc']:.2f}",
                    'KD权重': f"{result['best_config'].get('beta_kd', 0):.3f}",
                    'AFD权重': f"{result['best_config'].get('beta_afd', 0):.3f}",
                    'CRD权重': f"{result['best_config'].get('beta_crd', 0):.3f}",
                    '试验ID': result['best_trial_id']
                }
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # 保存为CSV
        if output_path:
            csv_path = output_path.replace('.png', '.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"结果表格已保存到: {csv_path}")
        
        # 创建可视化表格
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('BOHB超参数优化结果汇总', fontsize=16, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"结果表格图已保存到: {output_path}")
        plt.show()
        
        return df
    
    def generate_all_plots(self, output_dir=None):
        """生成所有图表"""
        if output_dir is None:
            output_dir = os.path.join(self.result_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        print("开始生成可视化图表...")
        
        # 1. 性能对比图
        self.plot_performance_comparison(
            os.path.join(output_dir, "performance_comparison.png")
        )
        
        # 2. 热力图
        self.plot_hyperparameter_heatmaps(
            os.path.join(output_dir, "heatmaps")
        )
        
        # 3. 收敛曲线
        self.plot_convergence_curve(
            os.path.join(output_dir, "convergence_curve.png")
        )
        
        # 4. 超参数分布
        self.plot_hyperparameter_distribution(
            os.path.join(output_dir, "hyperparameter_distribution.png")
        )
        
        # 5. 结果表格
        self.create_results_table(
            os.path.join(output_dir, "results_table.png")
        )
        
        print("所有图表生成完成！")

def main():
    parser = argparse.ArgumentParser(description='BOHB实验结果可视化')
    parser.add_argument('--result_dir', type=str, default='./save/bohb_results',
                       help='BOHB结果目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径')
    parser.add_argument('--all', action='store_true',
                       help='生成所有图表')
    parser.add_argument('--comparison', action='store_true',
                       help='生成性能对比图')
    parser.add_argument('--heatmaps', action='store_true',
                       help='生成热力图')
    parser.add_argument('--convergence', action='store_true',
                       help='生成收敛曲线')
    parser.add_argument('--distribution', action='store_true',
                       help='生成超参数分布图')
    parser.add_argument('--table', action='store_true',
                       help='生成结果表格')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = BOHBResultsVisualizer(args.result_dir)
    
    try:
        visualizer.load_results()
        print("结果加载成功！")
        
        if args.all or (not any([args.comparison, args.heatmaps, args.convergence, args.distribution, args.table])):
            # 生成所有图表
            visualizer.generate_all_plots(args.output_dir)
        else:
            # 生成指定图表
            output_dir = args.output_dir or os.path.join(args.result_dir, "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            if args.comparison:
                visualizer.plot_performance_comparison(
                    os.path.join(output_dir, "performance_comparison.png")
                )
            
            if args.heatmaps:
                visualizer.plot_hyperparameter_heatmaps(
                    os.path.join(output_dir, "heatmaps")
                )
            
            if args.convergence:
                visualizer.plot_convergence_curve(
                    os.path.join(output_dir, "convergence_curve.png")
                )
            
            if args.distribution:
                visualizer.plot_hyperparameter_distribution(
                    os.path.join(output_dir, "hyperparameter_distribution.png")
                )
            
            if args.table:
                visualizer.create_results_table(
                    os.path.join(output_dir, "results_table.png")
                )
                
    except Exception as e:
        print(f"错误: {e}")
        print("请确保BOHB结果目录包含有效的JSON结果文件")

if __name__ == "__main__":
    main()