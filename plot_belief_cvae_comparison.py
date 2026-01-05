import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
belief_files = [
    "checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-mlp-dense-remember-shape-color-3x2-v0-belief__42__rgb_joints_belief__20251228_053231/20251228_053231/training_metrics.csv",
    "checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/belief_baseline_33/20251228_015256/training_metrics.csv",
]

cvae_files = [
    "checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/cvae_belief__33__rgb_joints_belief__20260103_220033/cvae/training_metrics.csv",
    "checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/cvae_44/20260103_230807/training_metrics.csv",
]

def load_and_filter_eval(file_path):
    """Load CSV and filter for eval mode only."""
    df = pd.read_csv(file_path)
    eval_df = df[df['mode'] == 'eval'].copy()
    return eval_df

def average_across_seeds(file_list):
    """Load multiple seed files and average their metrics."""
    dfs = [load_and_filter_eval(f) for f in file_list]
    
    # Align by total_env_steps
    merged = dfs[0][['total_env_steps', 'success_once', 'success_at_end']].copy()
    merged = merged.rename(columns={'success_once': 'success_once_0', 'success_at_end': 'success_at_end_0'})
    
    for i, df in enumerate(dfs[1:], 1):
        temp = df[['total_env_steps', 'success_once', 'success_at_end']].copy()
        temp = temp.rename(columns={'success_once': f'success_once_{i}', 'success_at_end': f'success_at_end_{i}'})
        merged = pd.merge(merged, temp, on='total_env_steps', how='outer')
    
    # Calculate mean and std
    success_once_cols = [col for col in merged.columns if col.startswith('success_once_')]
    success_at_end_cols = [col for col in merged.columns if col.startswith('success_at_end_')]
    
    result = pd.DataFrame()
    result['total_env_steps'] = merged['total_env_steps']
    result['success_once_mean'] = merged[success_once_cols].mean(axis=1)
    result['success_once_std'] = merged[success_once_cols].std(axis=1)
    result['success_at_end_mean'] = merged[success_at_end_cols].mean(axis=1)
    result['success_at_end_std'] = merged[success_at_end_cols].std(axis=1)
    
    return result.sort_values('total_env_steps')

# Load and average data
print("Loading Belief baseline data...")
belief_data = average_across_seeds(belief_files)
print("Loading CVAE (Ours) data...")
cvae_data = average_across_seeds(cvae_files)

# Plotting style
try:
    plt.style.use('seaborn-whitegrid')
except:
    pass  # Use default style if seaborn style not available
fig_size = (10, 6)
colors = {'Belief': '#1f77b4', 'Ours (CVAE)': '#ff7f0e'}

# Plot 1: Eval Success Once
fig1, ax1 = plt.subplots(figsize=fig_size)
ax1.plot(belief_data['total_env_steps'], belief_data['success_once_mean'], 
         label='Belief', color=colors['Belief'], linewidth=2)
ax1.fill_between(belief_data['total_env_steps'], 
                  belief_data['success_once_mean'] - belief_data['success_once_std'],
                  belief_data['success_once_mean'] + belief_data['success_once_std'],
                  alpha=0.2, color=colors['Belief'])

ax1.plot(cvae_data['total_env_steps'], cvae_data['success_once_mean'], 
         label='Ours (CVAE)', color=colors['Ours (CVAE)'], linewidth=2)
ax1.fill_between(cvae_data['total_env_steps'], 
                  cvae_data['success_once_mean'] - cvae_data['success_once_std'],
                  cvae_data['success_once_mean'] + cvae_data['success_once_std'],
                  alpha=0.2, color=colors['Ours (CVAE)'])

ax1.set_xlabel('Steps', fontsize=12)
ax1.set_ylabel('Eval Success Once', fontsize=12)
ax1.set_title('RememberShapeAndColor3x2-v0 - Eval Success Once', fontsize=14)
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xlim(0, None)
ax1.set_ylim(0, 1)
plt.tight_layout()
fig1.savefig('RememberShapeAndColor3x2-v0_eval_success_once_comparison.png', dpi=150)
print("Saved: RememberShapeAndColor3x2-v0_eval_success_once_comparison.png")

# Plot 2: Eval Success At End
fig2, ax2 = plt.subplots(figsize=fig_size)
ax2.plot(belief_data['total_env_steps'], belief_data['success_at_end_mean'], 
         label='Belief', color=colors['Belief'], linewidth=2)
ax2.fill_between(belief_data['total_env_steps'], 
                  belief_data['success_at_end_mean'] - belief_data['success_at_end_std'],
                  belief_data['success_at_end_mean'] + belief_data['success_at_end_std'],
                  alpha=0.2, color=colors['Belief'])

ax2.plot(cvae_data['total_env_steps'], cvae_data['success_at_end_mean'], 
         label='Ours (CVAE)', color=colors['Ours (CVAE)'], linewidth=2)
ax2.fill_between(cvae_data['total_env_steps'], 
                  cvae_data['success_at_end_mean'] - cvae_data['success_at_end_std'],
                  cvae_data['success_at_end_mean'] + cvae_data['success_at_end_std'],
                  alpha=0.2, color=colors['Ours (CVAE)'])

ax2.set_xlabel('Steps', fontsize=12)
ax2.set_ylabel('Eval Success At End', fontsize=12)
ax2.set_title('RememberShapeAndColor3x2-v0 - Eval Success At End', fontsize=14)
ax2.legend(fontsize=11, loc='lower right')
ax2.set_xlim(0, None)
ax2.set_ylim(0, 1)
plt.tight_layout()
fig2.savefig('RememberShapeAndColor3x2-v0_eval_success_end_comparison.png', dpi=150)
print("Saved: RememberShapeAndColor3x2-v0_eval_success_end_comparison.png")

plt.show()
print("Done!")
