import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Remove if running in Jupyter/IDE

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
col_names = ['timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']

df_m = pd.read_csv('data/data_m.csv', comment='#', header=None, names=col_names)
df_v = pd.read_csv('data/data_v.csv', comment='#', header=None, names=col_names)

df_m['environment'] = 'Machine Hall'
df_v['environment'] = 'Vicon Room 1'

print("Datasets Loaded Successfully!\n")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
df = pd.concat([df_m, df_v], ignore_index=True)

# Convert timestamp ns → seconds
df['timestamp'] = df['timestamp'] / 1e9

# Align timestamps: shift each environment so it starts at Vicon Room 1's start
# (so both series overlap on the same x-axis with +1.403715e9 offset)
t_ref = df[df['environment'] == 'Vicon Room 1']['timestamp'].min()
for env in df['environment'].unique():
    mask = df['environment'] == env
    t_min_env = df.loc[mask, 'timestamp'].min()
    df.loc[mask, 'timestamp'] = df.loc[mask, 'timestamp'] - t_min_env + t_ref

# Compute noise magnitudes
df['gyro_noise'] = np.sqrt(df['w_x']**2 + df['w_y']**2 + df['w_z']**2)
df['acc_noise']  = np.sqrt(df['a_x']**2 + df['a_y']**2 + df['a_z']**2)

print("Preprocessing Completed")
print(df[['timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z',
          'gyro_noise', 'acc_noise', 'environment']].head())

# ─────────────────────────────────────────────
# 3. NOISE SUMMARY STATISTICS
# ─────────────────────────────────────────────
summary = df.groupby('environment').agg(
    gyro_mean = ('gyro_noise', 'mean'),
    gyro_std  = ('gyro_noise', 'std'),
    gyro_max  = ('gyro_noise', 'max'),
    gyro_min  = ('gyro_noise', 'min'),
    acc_mean  = ('acc_noise',  'mean'),
    acc_std   = ('acc_noise',  'std'),
    acc_max   = ('acc_noise',  'max'),
    acc_min   = ('acc_noise',  'min'),
).round(6)

print("\nNoise Summary Statistics:")
print(summary.to_string())

# ─────────────────────────────────────────────
# 4. PLOT 1 – Gyroscope Noise Boxplot
#    Y-axis: 0.0 to 1.0
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

groups = [df[df['environment'] == e]['gyro_noise'].values
          for e in ['Machine Hall', 'Vicon Room 1']]

bp = ax.boxplot(
    groups,
    labels=['Machine Hall', 'Vicon Room 1'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='none',
                    markeredgecolor='grey', markersize=3, alpha=0.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    boxprops=dict(linewidth=1.2)
)
for patch in bp['boxes']:
    patch.set_facecolor('#3a7ebf')

ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.set_title('Gyroscope Noise: Machine Hall vs Vicon Room 1')
ax.set_xlabel('environment')
ax.set_ylabel('Gyro Noise Magnitude')
plt.tight_layout()
plt.savefig('plot1_gyro_boxplot.png', dpi=150)
plt.close()
print("\nSaved: plot1_gyro_boxplot.png")

# ─────────────────────────────────────────────
# 5. PLOT 2 – Accelerometer Noise Boxplot
#    Y-axis: 4 to 17
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

groups_acc = [df[df['environment'] == e]['acc_noise'].values
              for e in ['Machine Hall', 'Vicon Room 1']]

bp2 = ax.boxplot(
    groups_acc,
    labels=['Machine Hall', 'Vicon Room 1'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='none',
                    markeredgecolor='grey', markersize=3, alpha=0.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    boxprops=dict(linewidth=1.2)
)
for patch in bp2['boxes']:
    patch.set_facecolor('#3a7ebf')

ax.set_ylim(4, 17)
ax.set_yticks([6, 8, 10, 12, 14, 16])
ax.set_title('Accelerometer Noise Comparison')
ax.set_xlabel('environment')
ax.set_ylabel('Accelerometer Noise Magnitude')
plt.tight_layout()
plt.savefig('plot2_acc_boxplot.png', dpi=150)
plt.close()
print("Saved: plot2_acc_boxplot.png")

# ─────────────────────────────────────────────
# 6. PLOT 3 – Noise Variation Over Time
#    X-axis: aligned timestamps with +1.403715e9 offset, range ~273 to ~460
#    Y-axis: 0.0 to 1.0
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

# Vicon Room 1 first (blue, underneath), Machine Hall on top (orange)
colors = {'Vicon Room 1': '#1f77b4', 'Machine Hall': '#ff7f0e'}

for env in ['Vicon Room 1', 'Machine Hall']:
    grp = df[df['environment'] == env].sort_values('timestamp')
    grp_ds = grp.iloc[::5]   # every 5th point for readability
    ax.plot(grp_ds['timestamp'], grp_ds['gyro_noise'],
            label=env, color=colors[env], linewidth=0.8, alpha=0.9)

ax.set_xlim(df['timestamp'].min(), df['timestamp'].max())
ax.set_ylim(0.0, 1.0)
# Force matplotlib to show the +1.403715e9 offset (same as original image)
ax.ticklabel_format(axis='x', style='sci', scilimits=(9, 9), useMathText=False)
ax.set_title('Noise Variation Over Time')
ax.set_xlabel('timestamp')
ax.set_ylabel('gyro_noise')
ax.legend(title='environment', loc='upper right')
plt.tight_layout()
plt.savefig('plot3_noise_over_time.png', dpi=150)
plt.close()
print("Saved: plot3_noise_over_time.png")

# ─────────────────────────────────────────────
# 7. PLOT 4 – Gyroscope Noise Density Heatmap (KDE)
#    X (w_x): -0.75 to 0.75 | Y (w_y): -0.6 to 0.7
# ─────────────────────────────────────────────
try:
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7, 6))

    xgrid = np.linspace(-0.85, 0.85, 250)
    ygrid = np.linspace(-0.70, 0.75, 250)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    cmaps = {'Machine Hall': 'Blues', 'Vicon Room 1': 'Oranges'}

    for env, cmap in cmaps.items():
        sub = df[df['environment'] == env][['w_x', 'w_y']].dropna()
        sub = sub.sample(min(3000, len(sub)), random_state=42)
        x, y = sub['w_x'].values, sub['w_y'].values
        kernel = gaussian_kde(np.vstack([x, y]), bw_method=0.15)
        Z = kernel(positions).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=12, cmap=cmap, alpha=0.6)
        ax.contour(X, Y, Z, levels=12,
                   colors='saddlebrown' if cmap == 'Oranges' else 'navy',
                   linewidths=0.4, alpha=0.35)

    legend_elements = [
        Patch(facecolor='#3a7ebf', label='Machine Hall'),
        Patch(facecolor='#ff8c00', label='Vicon Room 1'),
    ]
    ax.legend(handles=legend_elements, title='environment', loc='upper right')
    ax.set_xlim(-0.75, 0.80)
    ax.set_ylim(-0.65, 0.70)
    ax.set_xticks([-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75])
    ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_title('Gyroscope Noise Density Heatmap')
    ax.set_xlabel('w_x')
    ax.set_ylabel('w_y')
    plt.tight_layout()
    plt.savefig('plot4_gyro_density.png', dpi=150)
    plt.close()
    print("Saved: plot4_gyro_density.png")

except ImportError:
    print("scipy not installed — skipping KDE heatmap. Run: pip install scipy")

# ─────────────────────────────────────────────
# 8. PLOT 5 – Sensor Correlation Heatmap
#    Columns: w_x, w_y, w_z, a_x, a_y, a_z
#    Colormap: RdBu_r  |  Range: -1.0 to 1.0
# ─────────────────────────────────────────────
sensor_cols = ['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']
corr = df[sensor_cols].corr().round(3)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Axis ticks and labels
ax.set_xticks(range(len(sensor_cols)))
ax.set_yticks(range(len(sensor_cols)))
ax.set_xticklabels(sensor_cols)
ax.set_yticklabels(sensor_cols)

# Annotate each cell with its correlation value
for i in range(len(sensor_cols)):
    for j in range(len(sensor_cols)):
        val = corr.values[i, j]
        # Use white text on dark cells, black on light cells
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.3g}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='bold')

ax.set_title('Sensor Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot5_sensor_correlation.png', dpi=150)
plt.close()
print("Saved: plot5_sensor_correlation.png")

# ─────────────────────────────────────────────
# 9. SAVE PROCESSED DATASET
# ─────────────────────────────────────────────
df.to_csv('processed_imu_data.csv', index=False)
print("\nProcessed dataset saved successfully!")
