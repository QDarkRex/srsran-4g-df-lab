# Save as compare_amp.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_and_validate(filepath, name):
    """Load CSV and validate required columns"""
    try:
        df = pd.read_csv(filepath)
        required_cols = ['frame', 'mean_phase', 'phase_var', 'mean_mag', 'mag_var', 'drift_rate_rps']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Error: {name} missing columns: {missing}")
            return None
        print(f"✅ Loaded {name}: {len(df)} samples")
        return df
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def normalize_time_series(df, max_points=200):
    """
    Normalize time series to have consistent number of points for fair comparison
    This prevents skew when one dataset has many more samples than the other
    """
    if len(df) <= max_points:
        return df.copy()
    
    # Sample evenly across the time range
    indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    normalized_df = df.iloc[indices].copy()
    print(f"   Normalized from {len(df)} to {max_points} samples for fair comparison")
    return normalized_df

def calculate_stats(df, name):
    """Calculate statistics for a dataset"""
    stats = {
        'name': name,
        'samples': len(df),
        'phase_var_mean': df['phase_var'].mean(),
        'phase_var_std': df['phase_var'].std(),
        'phase_var_max': df['phase_var'].max(),
        'phase_var_min': df['phase_var'].min(),
        'phase_var_median': df['phase_var'].median(),
        'mag_var_mean': df['mag_var'].mean(),
        'mag_var_std': df['mag_var'].std(),
        'mag_var_max': df['mag_var'].max(),
        'mag_var_median': df['mag_var'].median(),
        'mean_mag_mean': df['mean_mag'].mean(),
        'mean_mag_std': df['mean_mag'].std(),
        'mean_mag_median': df['mean_mag'].median(),
        'drift_rate_mean': df['drift_rate_rps'].mean(),
        'drift_rate_std': df['drift_rate_rps'].std(),
        'drift_rate_abs_mean': df['drift_rate_rps'].abs().mean(),
        'drift_rate_abs_median': df['drift_rate_rps'].abs().median(),
    }
    return stats

def detect_surges_adaptive(df, threshold_multiplier=3):
    """
    Detect surge events - automatically adapts to data amount
    Uses percentile-based thresholds for data-agnostic detection
    """
    if len(df) < 5:
        df['is_surge'] = False
        df['surge_score'] = 0
        return df
    
    # Use percentiles instead of rolling windows for small datasets
    if len(df) < 20:
        # For small datasets, use global percentiles
        mag_high_threshold = df['mean_mag'].quantile(0.95)
        phase_high_threshold = df['phase_var'].quantile(0.95)
        
        df['mag_surge'] = df['mean_mag'] > mag_high_threshold
        df['phase_surge'] = df['phase_var'] > phase_high_threshold
        df['is_surge'] = df['mag_surge'] | df['phase_surge']
        
        # Calculate surge score (0-1)
        mag_max = df['mean_mag'].max()
        phase_max = df['phase_var'].max()
        df['surge_score'] = np.maximum(
            df['mean_mag'] / (mag_max + 1e-9),
            df['phase_var'] / (phase_max + 1e-9)
        )
    else:
        # For larger datasets, use rolling windows
        window = max(3, min(30, int(len(df) * 0.15)))
        
        rolling_mean_mag = df['mean_mag'].rolling(window=window, center=True, min_periods=1).mean()
        rolling_std_mag = df['mean_mag'].rolling(window=window, center=True, min_periods=1).std()
        rolling_std_mag = rolling_std_mag.replace(0, 1e-9)
        
        mag_surge_threshold = rolling_mean_mag + threshold_multiplier * rolling_std_mag
        df['mag_surge'] = df['mean_mag'] > mag_surge_threshold
        
        rolling_mean_phase = df['phase_var'].rolling(window=window, center=True, min_periods=1).mean()
        rolling_std_phase = df['phase_var'].rolling(window=window, center=True, min_periods=1).std()
        rolling_std_phase = rolling_std_phase.replace(0, 1e-9)
        
        phase_surge_threshold = rolling_mean_phase + threshold_multiplier * rolling_std_phase
        df['phase_surge'] = df['phase_var'] > phase_surge_threshold
        
        df['is_surge'] = df['mag_surge'] | df['phase_surge']
        
        mag_score = np.clip((df['mean_mag'] - rolling_mean_mag) / (rolling_std_mag + 1e-9) / threshold_multiplier, 0, 1)
        phase_score = np.clip((df['phase_var'] - rolling_mean_phase) / (rolling_std_phase + 1e-9) / threshold_multiplier, 0, 1)
        df['surge_score'] = np.maximum(mag_score, phase_score)
    
    df['is_surge'] = df['is_surge'].fillna(False)
    df['surge_score'] = df['surge_score'].fillna(0)
    
    return df

def analyze_surges(df_amp, df_without):
    """Analyze surge patterns in WITH AMPLIFIER data - data agnostic"""
    
    if len(df_amp) < 5:
        return None, None, None
    
    # Separate normal vs surge periods for WITH AMPLIFIER
    normal_amp = df_amp[~df_amp['is_surge']]
    surge_amp = df_amp[df_amp['is_surge']]
    
    # Find continuous surge blocks
    surge_blocks = []
    block_length = 0
    in_surge = False
    
    for idx, row in df_amp.iterrows():
        if row['is_surge']:
            if not in_surge:
                block_length = 1
                in_surge = True
            else:
                block_length += 1
        else:
            if in_surge:
                surge_blocks.append(block_length)
                in_surge = False
    
    if in_surge:
        surge_blocks.append(block_length)
    
    surge_stats = {
        'total_frames': len(df_amp),
        'total_surge_frames': len(surge_amp),
        'surge_percentage': (len(surge_amp) / len(df_amp)) * 100,
        'num_surge_blocks': len(surge_blocks),
        'avg_surge_duration': np.mean(surge_blocks) if surge_blocks else 0,
        'max_surge_duration': max(surge_blocks) if surge_blocks else 0,
        'median_surge_duration': np.median(surge_blocks) if surge_blocks else 0,
    }
    
    # Calculate statistics for NORMAL operation (WITH AMPLIFIER)
    if len(normal_amp) > 0:
        normal_stats = {
            'phase_var_mean': normal_amp['phase_var'].mean(),
            'phase_var_median': normal_amp['phase_var'].median(),
            'phase_var_std': normal_amp['phase_var'].std(),
            'mag_var_mean': normal_amp['mag_var'].mean(),
            'mag_var_median': normal_amp['mag_var'].median(),
            'mean_mag_mean': normal_amp['mean_mag'].mean(),
            'mean_mag_median': normal_amp['mean_mag'].median(),
            'drift_rate_abs_mean': normal_amp['drift_rate_rps'].abs().mean(),
            'drift_rate_abs_median': normal_amp['drift_rate_rps'].abs().median(),
        }
    else:
        normal_stats = None
    
    # Calculate statistics for SURGE periods (WITH AMPLIFIER)
    if len(surge_amp) > 0:
        surge_stats_metrics = {
            'phase_var_mean': surge_amp['phase_var'].mean(),
            'phase_var_median': surge_amp['phase_var'].median(),
            'phase_var_std': surge_amp['phase_var'].std(),
            'mag_var_mean': surge_amp['mag_var'].mean(),
            'mag_var_median': surge_amp['mag_var'].median(),
            'mean_mag_mean': surge_amp['mean_mag'].mean(),
            'mean_mag_median': surge_amp['mean_mag'].median(),
            'drift_rate_abs_mean': surge_amp['drift_rate_rps'].abs().mean(),
            'drift_rate_abs_median': surge_amp['drift_rate_rps'].abs().median(),
        }
    else:
        surge_stats_metrics = None
    
    return surge_stats, normal_stats, surge_stats_metrics

def print_header(title, char="="):
    """Print a formatted header"""
    print("\n" + char * 80)
    print(f" {title} ".center(80, char))
    print(char * 80)

def print_section(title, char="-"):
    """Print a section header"""
    print("\n" + char * 60)
    print(f" {title} ")
    print(char * 60)

def print_surge_analysis(surge_stats, normal_stats, surge_stats_metrics, stats_without):
    """Print surge analysis results - data agnostic"""
    
    print_header("🔍 SURGE DETECTION ANALYSIS (WITH AMPLIFIER DATA ONLY)")
    
    if surge_stats is None:
        print("Insufficient data for surge detection (<5 samples)")
        return
    
    normal_frames = surge_stats['total_frames'] - surge_stats['total_surge_frames']
    print(f"\n📊 SURGE STATISTICS (WITH AMPLIFIER):")
    print(f"   ├─ Total frames: {surge_stats['total_frames']}")
    print(f"   ├─ Normal frames: {normal_frames} ({normal_frames/surge_stats['total_frames']*100:.1f}%)")
    print(f"   ├─ Surge frames: {surge_stats['total_surge_frames']} ({surge_stats['surge_percentage']:.1f}%)")
    print(f"   ├─ Number of surge events: {surge_stats['num_surge_blocks']}")
    
    if surge_stats['avg_surge_duration'] > 0:
        print(f"   ├─ Average surge duration: {surge_stats['avg_surge_duration']:.1f} frames")
        print(f"   ├─ Max surge duration: {surge_stats['max_surge_duration']} frames")
        print(f"   └─ Median surge duration: {surge_stats['median_surge_duration']:.1f} frames")
    
    if normal_stats and surge_stats_metrics:
        print_section("📈 WITH AMPLIFIER: NORMAL vs SURGE PERIODS")
        print(f"\n{'Metric':<35} {'🔵 NORMAL':<18} {'🔴 SURGE':<18} {'📊 RATIO':<12}")
        print("-" * 85)
        
        phase_ratio = surge_stats_metrics['phase_var_mean'] / (normal_stats['phase_var_mean'] + 1e-9)
        print(f"{'Phase Variance (mean)':<35} {normal_stats['phase_var_mean']:<18.4f} {surge_stats_metrics['phase_var_mean']:<18.4f} {phase_ratio:<12.1f}x")
        
        mag_ratio = surge_stats_metrics['mag_var_mean'] / (normal_stats['mag_var_mean'] + 1e-9)
        print(f"{'Magnitude Variance (mean)':<35} {normal_stats['mag_var_mean']:<18.4f} {surge_stats_metrics['mag_var_mean']:<18.4f} {mag_ratio:<12.1f}x")
        
        mean_mag_ratio = surge_stats_metrics['mean_mag_mean'] / (normal_stats['mean_mag_mean'] + 1e-9)
        print(f"{'Mean Magnitude (mean)':<35} {normal_stats['mean_mag_mean']:<18.2f} {surge_stats_metrics['mean_mag_mean']:<18.2f} {mean_mag_ratio:<12.1f}x")
        
        drift_diff = surge_stats_metrics['drift_rate_abs_mean'] - normal_stats['drift_rate_abs_mean']
        print(f"{'Drift Rate (abs mean)':<35} {normal_stats['drift_rate_abs_mean']:<18.4f} {surge_stats_metrics['drift_rate_abs_mean']:<18.4f} {drift_diff:<+12.4f}")
    
    # Diagnosis (using percentages which are inherently data-agnostic)
    print_section("🔍 SURGE DIAGNOSIS (WITH AMPLIFIER)")
    if surge_stats['surge_percentage'] > 20:
        print("   🔴 SEVERE: >20% of WITH AMPLIFIER data is surge events")
        print("      → Amplifier likely oscillating or severely unstable")
    elif surge_stats['surge_percentage'] > 5:
        print("   🟡 MODERATE: 5-20% of WITH AMPLIFIER data is surge events")
        print("      → Intermittent issue with amplifier")
    elif surge_stats['surge_percentage'] > 0:
        print("   🟢 MINOR: <5% of WITH AMPLIFIER data is surge events")
        print("      → Occasional glitches, may be acceptable")
    else:
        print("   ✅ No surge events detected in WITH AMPLIFIER data")
    
    # Compare amplifier normal operation to baseline (WITHOUT AMPLIFIER)
    if normal_stats and stats_without:
        print_section("📊 AMPLIFIER PERFORMANCE (Normal Periods vs Baseline)")
        
        gain_ratio = normal_stats['mean_mag_mean'] / (stats_without['mean_mag_mean'] + 1e-9)
        gain_db = 10 * np.log10(gain_ratio)
        
        print(f"\n   WITHOUT Amplifier baseline mean magnitude: {stats_without['mean_mag_mean']:.2f}")
        print(f"   WITH Amplifier (normal periods) mean magnitude: {normal_stats['mean_mag_mean']:.2f}")
        print(f"\n   📈 Signal Gain: {gain_ratio:.2f}x ({gain_db:.1f} dB)")
        
        if gain_ratio < 0.8:
            print("   ❌ RESULT: Amplifier is ATTENUATING the signal!")
        elif gain_ratio < 5:
            print("   ⚠️  RESULT: Low gain ({:.1f} dB)".format(gain_db))
        else:
            print("   ✅ RESULT: Good gain ({:.1f} dB)".format(gain_db))
        
        # Compare phase noise
        phase_ratio_vs_baseline = normal_stats['phase_var_mean'] / (stats_without['phase_var_mean'] + 1e-9)
        print(f"\n   WITHOUT Amplifier phase variance: {stats_without['phase_var_mean']:.4f}")
        print(f"   WITH Amplifier (normal) phase variance: {normal_stats['phase_var_mean']:.4f}")
        print(f"   📊 Phase noise increase: {phase_ratio_vs_baseline:.1f}x")

def plot_comparison(df_without, df_amp, stats_without, stats_with_amp):
    """Create comparison plots - data amount agnostic with normalized axes"""
    
    # Normalize both datasets to the same number of points for fair visual comparison
    max_points = min(200, max(len(df_without), len(df_amp)))
    df_without_norm = normalize_time_series(df_without, max_points)
    df_amp_norm = normalize_time_series(df_amp, max_points)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('AMPLIFIER COMPARISON: WITHOUT (GREEN) vs WITH (RED)\n(Normalized for fair comparison)', 
                 fontsize=12, fontweight='bold')
    
    # Plot 1: Phase variance over time (normalized x-axis)
    x_without = np.linspace(0, 100, len(df_without_norm))
    x_amp = np.linspace(0, 100, len(df_amp_norm))
    
    axes[0,0].plot(x_without, df_without_norm['phase_var'], 'g-', alpha=0.7, label='⚫ WITHOUT Amplifier', linewidth=1.5)
    axes[0,0].plot(x_amp, df_amp_norm['phase_var'], 'r-', alpha=0.7, label='🔴 WITH Amplifier', linewidth=1.5)
    axes[0,0].set_title('Phase Variance Comparison (normalized time)')
    axes[0,0].set_xlabel('Normalized Time (%)')
    axes[0,0].set_ylabel('Variance (rad²)')
    axes[0,0].legend(loc='upper left', fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Plot 2: Mean magnitude over time (normalized)
    axes[0,1].plot(x_without, df_without_norm['mean_mag'], 'g-', alpha=0.7, label='⚫ WITHOUT Amplifier', linewidth=1.5)
    axes[0,1].plot(x_amp, df_amp_norm['mean_mag'], 'r-', alpha=0.7, label='🔴 WITH Amplifier', linewidth=1.5)
    axes[0,1].set_title('Signal Strength Comparison (normalized time)')
    axes[0,1].set_xlabel('Normalized Time (%)')
    axes[0,1].set_ylabel('Mean Magnitude')
    axes[0,1].legend(loc='upper left', fontsize=9)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison (uses medians, not affected by sample size)
    box_data = [
        df_without['phase_var'].values,
        df_amp['phase_var'].values,
        df_without['mean_mag'].values,
        df_amp['mean_mag'].values
    ]
    
    bp = axes[0,2].boxplot(box_data, positions=[1, 2, 4, 5], widths=0.6,
                           patch_artist=True,
                           boxprops=dict(facecolor='lightgray'),
                           medianprops=dict(color='black', linewidth=2))
    
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)
    bp['boxes'][2].set_facecolor('green')
    bp['boxes'][2].set_alpha(0.5)
    bp['boxes'][3].set_facecolor('red')
    bp['boxes'][3].set_alpha(0.5)
    
    axes[0,2].set_xticklabels(['Phase Var\nW/O Amp', 'Phase Var\nW/ Amp', 
                               'Magnitude\nW/O Amp', 'Magnitude\nW/ Amp'])
    axes[0,2].set_title('Distribution Comparison (Median = middle line)')
    axes[0,2].set_ylabel('Value')
    axes[0,2].set_yscale('log')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Violin plot for better distribution visualization
    violin_data = [df_without['phase_var'].values, df_amp['phase_var'].values]
    parts = axes[1,0].violinplot(violin_data, positions=[1, 2], showmedians=True)
    parts['bodies'][0].set_facecolor('green')
    parts['bodies'][0].set_alpha(0.5)
    parts['bodies'][1].set_facecolor('red')
    parts['bodies'][1].set_alpha(0.5)
    axes[1,0].set_xticks([1, 2])
    axes[1,0].set_xticklabels(['🟢 WITHOUT Amp', '🔴 WITH Amp'])
    axes[1,0].set_title('Phase Variance Distribution (width = density)')
    axes[1,0].set_ylabel('Phase Variance (rad²)')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot with density coloring
    sample_without = df_without.iloc[::max(1, len(df_without)//300)]
    sample_amp = df_amp.iloc[::max(1, len(df_amp)//300)]
    
    axes[1,1].scatter(sample_without['mean_mag'], sample_without['phase_var'], 
                     alpha=0.5, s=15, c='green', label='⚫ WITHOUT Amp')
    axes[1,1].scatter(sample_amp['mean_mag'], sample_amp['phase_var'], 
                     alpha=0.5, s=15, c='red', label='🔴 WITH Amp')
    axes[1,1].set_title('Phase vs Magnitude Correlation')
    axes[1,1].set_xlabel('Mean Magnitude')
    axes[1,1].set_ylabel('Phase Variance (rad²)')
    axes[1,1].set_yscale('log')
    axes[1,1].legend(loc='upper left', fontsize=9)
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Statistical comparison bar chart (uses medians for robustness)
    metrics = ['Phase Var\n(median)', 'Mag Var\n(median)', 'Drift Rate\n(median)']
    without_vals = [
        stats_without['phase_var_median'],
        stats_without['mag_var_median'],
        stats_without['drift_rate_abs_median']
    ]
    with_vals = [
        stats_with_amp['phase_var_median'],
        stats_with_amp['mag_var_median'],
        stats_with_amp['drift_rate_abs_median']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1,2].bar(x - width/2, without_vals, width, label='🟢 WITHOUT Amp', color='green', alpha=0.7)
    bars2 = axes[1,2].bar(x + width/2, with_vals, width, label='🔴 WITH Amp', color='red', alpha=0.7)
    
    axes[1,2].set_ylabel('Value (log scale)')
    axes[1,2].set_title('Median Comparison (Robust to outliers)')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(metrics)
    axes[1,2].legend()
    axes[1,2].set_yscale('log')
    axes[1,2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def main():
    print_header("🔬 AMPLIFIER COMPARISON ANALYSIS TOOL (Data-Agnostic)", "=")
    print("\nThis tool compares system performance:")
    print("   🟢 WITHOUT Amplifier (baseline)")
    print("   🔴 WITH Amplifier (measurement)")
    print("\n   📊 All comparisons are normalized and use medians where appropriate")
    print("   📊 Percentages and ratios are used instead of absolute values")
    
    # Check command line arguments or use defaults
    if len(sys.argv) == 3:
        file_without = sys.argv[1]
        file_with_amp = sys.argv[2]
    else:
        # Default paths
        file_without = '/tmp/amplifier_analysis_wo_amplifier.csv'
        file_with_amp = '/tmp/amplifier_analysis.csv'
        
        if not os.path.exists(file_without):
            print(f"\n❌ File not found: {file_without}")
            print("\nUsage: python compare_amp.py <without_amp.csv> <with_amp.csv>")
            sys.exit(1)
        if not os.path.exists(file_with_amp):
            print(f"\n❌ File not found: {file_with_amp}")
            sys.exit(1)
    
    print(f"\n📂 Loading data...")
    print(f"   🟢 WITHOUT Amplifier: {file_without}")
    print(f"   🔴 WITH Amplifier: {file_with_amp}")
    
    # Load both datasets
    df_without = load_and_validate(file_without, "🟢 WITHOUT Amplifier")
    df_with_amp = load_and_validate(file_with_amp, "🔴 WITH Amplifier")
    
    if df_without is None or df_with_amp is None:
        sys.exit(1)

    # An empty-but-valid CSV (right columns, zero rows) is not a "no data" error
    # by pandas' standards, but every stat below becomes NaN, and NaN < X is
    # always False in Python — so the verdict logic would silently fall through
    # to the final "else" branch and print a clean bill of health. Fail loudly
    # instead of risking a false "amplifier is fine" verdict on missing data.
    if len(df_without) == 0:
        print(f"\n❌ Error: {file_without} has the right columns but 0 data rows. Nothing to compare.")
        sys.exit(1)
    if len(df_with_amp) == 0:
        print(f"\n❌ Error: {file_with_amp} has the right columns but 0 data rows. Nothing to compare.")
        sys.exit(1)

    # Detect surges in WITH AMPLIFIER data only
    print("\n🔍 Analyzing WITH Amplifier data for surge events...")
    df_with_amp = detect_surges_adaptive(df_with_amp)
    
    # Calculate statistics for both
    stats_without = calculate_stats(df_without, "🟢 WITHOUT Amplifier")
    stats_with_amp = calculate_stats(df_with_amp, "🔴 WITH Amplifier")
    
    # Analyze surges
    surge_stats, normal_stats, surge_metrics = analyze_surges(df_with_amp, df_without)
    
    # ========================================================================
    # PRINT COMPARISON RESULTS (Using medians for robustness)
    # ========================================================================
    
    print_header("📊 DIRECT COMPARISON: WITHOUT vs WITH AMPLIFIER")
    
    print(f"\n{'Metric':<40} {'🟢 WITHOUT Amp':<22} {'🔴 WITH Amp':<22} {'📊 RATIO':<12}")
    print("-" * 100)
    
    # Phase variance (using median for robustness)
    phase_ratio = stats_with_amp['phase_var_median'] / (stats_without['phase_var_median'] + 1e-9)
    phase_arrow = "⬆️" if phase_ratio > 1 else "⬇️"
    print(f"{'Phase Variance (median)':<40} {stats_without['phase_var_median']:<22.6f} {stats_with_amp['phase_var_median']:<22.6f} {phase_ratio:<11.2f}x {phase_arrow}")
    
    # Magnitude variance (using median)
    mag_ratio = stats_with_amp['mag_var_median'] / (stats_without['mag_var_median'] + 1e-9)
    mag_arrow = "⬆️" if mag_ratio > 1 else "⬇️"
    print(f"{'Magnitude Variance (median)':<40} {stats_without['mag_var_median']:<22.6f} {stats_with_amp['mag_var_median']:<22.6f} {mag_ratio:<11.2f}x {mag_arrow}")
    
    # Mean magnitude - SIGNAL STRENGTH (using median)
    mag_gain = stats_with_amp['mean_mag_median'] / (stats_without['mean_mag_median'] + 1e-9)
    if mag_gain > 1:
        gain_arrow = "🔺"
        gain_text = f"{mag_gain:.2f}x GAIN"
    else:
        gain_arrow = "🔻"
        gain_text = f"{mag_gain:.2f}x LOSS"
    print(f"{'Mean Magnitude (median)':<40} {stats_without['mean_mag_median']:<22.2f} {stats_with_amp['mean_mag_median']:<22.2f} {mag_gain:<11.2f}x {gain_arrow} {gain_text}")
    
    # Drift rate (using median)
    drift_ratio = stats_with_amp['drift_rate_abs_median'] / (stats_without['drift_rate_abs_median'] + 1e-9)
    drift_arrow = "⬆️" if drift_ratio > 1 else "⬇️"
    print(f"{'Drift Rate (abs median)':<40} {stats_without['drift_rate_abs_median']:<22.4f} {stats_with_amp['drift_rate_abs_median']:<22.4f} {drift_ratio:<11.2f}x {drift_arrow}")
    
    # Sample info (for transparency)
    print(f"{'Sample count':<40} {stats_without['samples']:<22} {stats_with_amp['samples']:<22}")
    print(f"{'Sample difference':<40} {'':<22} {abs(stats_without['samples'] - stats_with_amp['samples']):<22} frames")
    
    # ========================================================================
    # PRINT DEGRADATION ANALYSIS
    # ========================================================================
    
    print_header("📉 DEGRADATION ANALYSIS")
    
    phase_noise_db = 10 * np.log10(phase_ratio)
    print(f"\n   Phase noise increase: {phase_noise_db:+.1f} dB ({phase_ratio:.1f}x)")
    print(f"   Magnitude instability: {10 * np.log10(mag_ratio):+.1f} dB ({mag_ratio:.1f}x)")
    
    # Determine overall verdict
    print_header("🎯 VERDICT", "=")

    # Belt-and-suspenders: a NaN here (e.g. a column with unparseable/missing
    # values mixed into otherwise-valid rows) compares as False against every
    # threshold below, which would otherwise fall through to the "acceptable"
    # branch instead of surfacing that the numbers are meaningless.
    if not (np.isfinite(mag_gain) and np.isfinite(phase_ratio)):
        print("\n   ⚠️  VERDICT: CANNOT DETERMINE — mag_gain or phase_ratio is NaN/inf.")
        print("      Check the input CSVs for missing/unparseable values in")
        print("      mean_mag / phase_var before trusting any comparison above.")
        sys.exit(1)

    if mag_gain < 0.8:
        print("\n   🔴 VERDICT: AMPLIFIER IS ATTENUATING (NOT AMPLIFYING)")
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   📋 ACTION ITEMS:")
        print("      1. Check if amplifier is POWERED ON")
        print("      2. Verify input/output orientation (reverse connection = loss)")
        print("      3. Test amplifier with known signal source")
    elif mag_gain < 5:
        print("\n   🟡 VERDICT: LOW AMPLIFIER GAIN ({:.1f} dB)".format(10*np.log10(mag_gain)))
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   📋 ACTION ITEMS:")
        print("      1. Expected gain for LNA: 20-30 dB (10-30x)")
        print("      2. Check if amplifier is underpowered")
        print("      3. Verify amplifier frequency matches your band")
    elif phase_ratio > 5:
        print("\n   🟡 VERDICT: AMPLIFIER ADDS SIGNIFICANT PHASE NOISE")
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   📋 ACTION ITEMS:")
        print("      1. Add phase noise filtering in software")
        print("      2. Improve amplifier power supply decoupling")
    else:
        print("\n   ✅ VERDICT: AMPLIFIER IS WORKING ACCEPTABLY")
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   📋 NOTES:")
        print("      • Gain: {:.1f} dB".format(10*np.log10(mag_gain)))
        print("      • Phase noise increase: {:.1f} dB".format(phase_noise_db))
    
    # Print surge analysis
    print_surge_analysis(surge_stats, normal_stats, surge_metrics, stats_without)
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    
    print_header("📈 GENERATING COMPARISON PLOTS")
    print("\n   Creating visual comparison with normalized axes...")
    
    fig = plot_comparison(df_without, df_with_amp, stats_without, stats_with_amp)
    plt.savefig('/tmp/amp_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ Plot saved to: /tmp/amp_comparison.png")
    
    # ========================================================================
    # FINAL SUMMARY TABLE (Using medians)
    # ========================================================================
    
    print_header("📋 FINAL SUMMARY TABLE (Using Medians - Robust)")
    
    summary_data = {
        'Metric': ['Phase Variance', 'Magnitude Variance', 'Signal Strength', 'Drift Rate (abs)', 'Samples'],
        '🟢 WITHOUT Amp': [
            f"{stats_without['phase_var_median']:.4f}",
            f"{stats_without['mag_var_median']:.4f}",
            f"{stats_without['mean_mag_median']:.2f}",
            f"{stats_without['drift_rate_abs_median']:.4f} rad/s",
            f"{stats_without['samples']}"
        ],
        '🔴 WITH Amp': [
            f"{stats_with_amp['phase_var_median']:.4f}",
            f"{stats_with_amp['mag_var_median']:.4f}",
            f"{stats_with_amp['mean_mag_median']:.2f}",
            f"{stats_with_amp['drift_rate_abs_median']:.4f} rad/s",
            f"{stats_with_amp['samples']}"
        ],
        'Change': [
            f"{phase_ratio:.1f}x",
            f"{mag_ratio:.1f}x",
            f"{mag_gain:.1f}x",
            f"{drift_ratio:.1f}x",
            f"{abs(stats_without['samples'] - stats_with_amp['samples'])} diff"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    print_header("✅ ANALYSIS COMPLETE", "=")
    print("\n   📁 Report saved to: /tmp/amp_comparison.png")
    print("\n   💡 Key improvements for data-agnostic analysis:")
    print("      • Using MEDIANS instead of means (robust to outliers)")
    print("      • Normalized time axes for fair visual comparison")
    print("      • Percentages for surge detection (not absolute counts)")
    print("      • Box/violin plots show distribution shape regardless of sample size")
    print("      • Sample size shown for transparency\n")

if __name__ == "__main__":
    main()