import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(case_path):
    ae_data64 = pd.read_csv(f'{case_path}/ae_data64.csv')
    ae_data94 = pd.read_csv(f'{case_path}/ae_data94.csv')
    ae_data124 = pd.read_csv(f'{case_path}/ae_data124.csv')
    ae_data154 = pd.read_csv(f'{case_path}/ae_data154.csv')

    re_data64 = pd.read_csv(f'{case_path}/re_data64.csv')
    re_data94 = pd.read_csv(f'{case_path}/re_data94.csv')
    re_data124 = pd.read_csv(f'{case_path}/re_data124.csv')
    re_data154 = pd.read_csv(f'{case_path}/re_data154.csv')

    # Convert RE data to percentages
    re_data64['RE'] *= 100
    re_data94['RE'] *= 100
    re_data124['RE'] *= 100
    re_data154['RE'] *= 100

    return ae_data64, ae_data94, ae_data124, ae_data154, re_data64, re_data94, re_data124, re_data154

def plot_histograms(case_name, ae_data64, ae_data94, ae_data124, ae_data154, re_data64, re_data94, re_data124, re_data154, ax_ae, ax_re):
    ae_bins = [0, 1.5, 3, 4.5, np.inf]
    re_bins = [0, 0.5, 1, 1.5, np.inf]

    ae_hist64, _ = np.histogram(ae_data64['AE'], bins=ae_bins)
    ae_hist94, _ = np.histogram(ae_data94['AE'], bins=ae_bins)
    ae_hist124, _ = np.histogram(ae_data124['AE'], bins=ae_bins)
    ae_hist154, _ = np.histogram(ae_data154['AE'], bins=ae_bins)

    re_hist64, _ = np.histogram(re_data64['RE'], bins=re_bins)
    re_hist94, _ = np.histogram(re_data94['RE'], bins=re_bins)
    re_hist124, _ = np.histogram(re_data124['RE'], bins=re_bins)
    re_hist154, _ = np.histogram(re_data154['RE'], bins=re_bins)

    colors = ['dodgerblue', 'orange', 'limegreen', 'red']
    bar_width = 0.15

    r1 = np.arange(len(ae_hist64))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    ax_ae.bar(r1, ae_hist64, color=colors[0], width=bar_width, edgecolor='grey')
    ax_ae.bar(r2, ae_hist94, color=colors[1], width=bar_width, edgecolor='grey')
    ax_ae.bar(r3, ae_hist124, color=colors[2], width=bar_width, edgecolor='grey')
    ax_ae.bar(r4, ae_hist154, color=colors[3], width=bar_width, edgecolor='grey')

    ax_ae.set_xlabel('Absolute Error of Temperature (K)', fontweight='bold', fontsize=12, fontname='Arial')
    ax_ae.set_ylabel('Number of Points', fontweight='bold', fontsize=12, fontname='Arial')
    ax_ae.set_title(f'AE - {case_name}', fontweight='bold', fontsize=14, fontname='Arial')
    ax_ae.set_xticks([r + 1.5 * bar_width for r in range(len(ae_hist64))])
    ax_ae.set_xticklabels(['a', 'b', 'c', 'd'], fontsize=12, fontname='Arial')
    ax_ae.tick_params(axis='both', which='major', labelsize=12)
    ax_ae.grid(True, linestyle='--', alpha=0.7)

    ax_re.bar(r1, re_hist64, color=colors[0], width=bar_width, edgecolor='grey')
    ax_re.bar(r2, re_hist94, color=colors[1], width=bar_width, edgecolor='grey')
    ax_re.bar(r3, re_hist124, color=colors[2], width=bar_width, edgecolor='grey')
    ax_re.bar(r4, re_hist154, color=colors[3], width=bar_width, edgecolor='grey')

    ax_re.set_xlabel('Relative Error of Temperature (%)', fontweight='bold', fontsize=12, fontname='Arial')
    ax_re.set_ylabel('Number of Points', fontweight='bold', fontsize=12, fontname='Arial')
    ax_re.set_title(f'RE - {case_name}', fontweight='bold', fontsize=14, fontname='Arial')
    ax_re.set_xticks([r + 1.5 * bar_width for r in range(len(re_hist64))])
    ax_re.set_xticklabels(['a', 'b', 'c', 'd'], fontsize=12, fontname='Arial')
    ax_re.tick_params(axis='both', which='major', labelsize=12)
    ax_re.grid(True, linestyle='--', alpha=0.7)

# Load data for all cases
case1_path = r'D:\桌面\case1AE与RE'
case2_path = r'D:\桌面\case2AE与RE'
case3_path = r'D:\桌面\case3AE与RE'

ae_data64_case1, ae_data94_case1, ae_data124_case1, ae_data154_case1, re_data64_case1, re_data94_case1, re_data124_case1, re_data154_case1 = load_data(case1_path)
ae_data64_case2, ae_data94_case2, ae_data124_case2, ae_data154_case2, re_data64_case2, re_data94_case2, re_data124_case2, re_data154_case2 = load_data(case2_path)
ae_data64_case3, ae_data94_case3, ae_data124_case3, ae_data154_case3, re_data64_case3, re_data94_case3, re_data124_case3, re_data154_case3 = load_data(case3_path)

# Create a new figure to arrange all subplots in a 2x3 grid
fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches (landscape)

# Create the main subplots
ax1_ae = fig.add_subplot(2, 3, 1)
ax1_re = fig.add_subplot(2, 3, 4)
ax2_ae = fig.add_subplot(2, 3, 2)
ax2_re = fig.add_subplot(2, 3, 5)
ax3_ae = fig.add_subplot(2, 3, 3)
ax3_re = fig.add_subplot(2, 3, 6)

# Generate plots for all cases and add them to the grid
plot_histograms('Case 1', ae_data64_case1, ae_data94_case1, ae_data124_case1, ae_data154_case1, re_data64_case1, re_data94_case1, re_data124_case1, re_data154_case1, ax1_ae, ax1_re)
plot_histograms('Case 2', ae_data64_case2, ae_data94_case2, ae_data124_case2, ae_data154_case2, re_data64_case2, re_data94_case2, re_data124_case2, re_data154_case2, ax2_ae, ax2_re)
plot_histograms('Case 3', ae_data64_case3, ae_data94_case3, ae_data124_case3, ae_data154_case3, re_data64_case3, re_data94_case3, re_data124_case3, re_data154_case3, ax3_ae, ax3_re)

# Adjust layout
plt.tight_layout()

# Create a single legend
colors = ['dodgerblue', 'orange', 'limegreen', 'red']
labels = ['PINN trained with 64 points', 'PINN trained with 94 points', 'PINN trained with 124 points', 'PINN trained with 154 points']

# Create a legend outside the plots
fig.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(colors))],
           labels=labels, loc='lower center', fontsize=12, ncol=4, bbox_to_anchor=(0.5, -0.05))

# Save the figure as an SVG file
output_path = r'D:\桌面\combined_histogramsnn11111.svg'
plt.savefig(output_path, format='svg', bbox_inches='tight')

plt.show()
