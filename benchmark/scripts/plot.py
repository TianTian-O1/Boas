import pandas as pd
import matplotlib.pyplot as plt

def plot_results():
    # 读取结果数据
    df = pd.read_csv('../results/results.csv')
    
    # 设置图表风格
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 创建图表
    plt.figure(figsize=(12, 7))
    
    # 为每个实现创建线图
    for i, implementation in enumerate(df['language'].unique()):
        data = df[df['language'] == implementation]
        if implementation == 'Boas':
            # Boas 使用实线，加粗显示
            plt.plot(data['size'], data['time_ms'], 
                    marker='o', 
                    markersize=8, 
                    label=implementation,
                    color='#ff7f0e',  # 使用醒目的橙色
                    linewidth=3)
        else:
            # 其他实现使用虚线
            plt.plot(data['size'], data['time_ms'], 
                    marker='o', 
                    markersize=6, 
                    label=implementation,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    linestyle='--',
                    alpha=0.7)
    
    # 设置标题和标签
    plt.title('Matrix Multiplication Performance Comparison\n(Lower is Better)', 
             fontsize=14, pad=20)
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.yscale('log')
    
    # 自定义图例
    plt.legend(title='Implementation', 
              title_fontsize=12, 
              fontsize=10, 
              loc='upper left')
    
    # 添加网格
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('../results/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 打印性能统计数据
    print("\nPerformance Statistics:")
    print("\nExecution Time (ms) by Matrix Size:")
    
    # 创建性能统计表格
    stats = df.pivot_table(
        values='time_ms',
        index='language',
        columns='size',
        aggfunc='mean'
    ).round(2)
    
    print(stats)
    
    # 计算相对性能（以 NumPy 为基准）
    if 'NumPy' in stats.index:
        print("\nRelative Performance (compared to NumPy):")
        numpy_times = stats.loc['NumPy']
        relative_perf = stats.div(numpy_times)
        print(relative_perf.round(2))

if __name__ == '__main__':
    plot_results()