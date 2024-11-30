import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results():
    # 读取结果
    df = pd.read_csv('../results/results.csv', encoding='utf-8')
    
    # 创建性能对比图
    plt.figure(figsize=(10, 6))
    
    # 设置样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 绘制时间对比图
    sns.lineplot(data=df, x='size', y='time_ms', hue='language', marker='o')
    
    # 设置标题和标签
    plt.title('Matrix Multiplication Performance Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    
    # 添加图例
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('../results/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印性能统计
    print("\nPerformance Statistics:")
    print("\nExecution Time (ms) by Matrix Size:")
    print(df.pivot_table(
        values='time_ms',
        index='language',
        columns='size',
        aggfunc='mean'
    ).round(2))

if __name__ == '__main__':
    plot_results()