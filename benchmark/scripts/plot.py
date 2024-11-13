import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_results():
    # 读取结果
    df = pd.read_csv('../results/results.csv')
    
    # 设置风格和颜色
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色和蓝色
    sns.set_palette(colors)
    
    # 设置字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # 创建性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 时间对比
    sns.lineplot(data=df, x='size', y='time_ms', hue='language', 
                marker='o', ax=ax1, linewidth=3, markersize=10)
    ax1.set_title('Execution Time Comparison', fontsize=16, pad=20, fontweight='bold')
    ax1.set_xlabel('Matrix Size', fontsize=14)
    ax1.set_ylabel('Time (ms)', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(title='Language', title_fontsize=14, fontsize=12, 
              bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    # 内存对比
    sns.lineplot(data=df, x='size', y='memory_kb', hue='language', 
                marker='o', ax=ax2, linewidth=3, markersize=10)
    ax2.set_title('Memory Usage Comparison', fontsize=16, pad=20, fontweight='bold')
    ax2.set_xlabel('Matrix Size', fontsize=14)
    ax2.set_ylabel('Memory (KB)', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(title='Language', title_fontsize=14, fontsize=12,
              bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    # 添加数据标签，使用科学计数法
    for ax in [ax1, ax2]:
        for line in ax.lines:
            if line.get_label() in df['language'].unique():
                y_data = line.get_ydata()
                x_data = line.get_xdata()
                for x, y in zip(x_data, y_data):
                    if not np.isnan(y):
                        if y >= 1000:
                            label = f'{y:.1e}'
                        else:
                            label = f'{y:.1f}'
                        ax.annotate(label, 
                                  (x, y), 
                                  textcoords="offset points",
                                  xytext=(0,10), 
                                  ha='center',
                                  fontsize=10,
                                  bbox=dict(facecolor='white', 
                                          edgecolor='none',
                                          alpha=0.7,
                                          pad=1))
    
    # 设置背景色和边框
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 为底部的图例留出空间
    
    # 保存图表
    plt.savefig('../results/comparison.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    # 打印详细结果表格
    print("\nDetailed Results:")
    pd.set_option('display.float_format', lambda x: '%.2f' if x < 1000 else '%.2e' % x)
    print(df.to_string(index=False))

if __name__ == '__main__':
    plot_results()