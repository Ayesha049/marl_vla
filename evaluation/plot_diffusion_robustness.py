import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_data(csv_file, output_file="robustness_plot.png"):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    # Load the data
    df = pd.read_csv(csv_file)

    # Columns to plot
    category_col = 'method' 
    x_col = 'noise_std'
    y_col = 'success_rate'

    plt.figure(figsize=(10, 6))
    
    if category_col in df.columns:
        for name, group in df.groupby(category_col):
            # Sort values by noise_std to ensure the line plots continuously from left to right
            group = group.sort_values(by=x_col)
            plt.plot(group[x_col], group[y_col], marker='o', linewidth=2, label=str(name))

    plt.xlabel('Noise Std', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate vs Noise Std', fontsize=14)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(PROJECT_ROOT, 'results', 'lift', 'robustness_eval_lift_diffusion.csv')
    
    output_file = csv_file.replace('.csv', '.png')
    plot_data(csv_file, output_file)