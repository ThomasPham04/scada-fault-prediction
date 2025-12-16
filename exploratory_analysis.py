"""
Exploratory Data Analysis: Feature Behavior Before Anomaly Events

This script analyzes how different sensor groups behave as anomaly events approach.
Groups features by:
- Temperature sensors
- Wind sensors  
- Power sensors
- Electrical sensors
- Mechanical sensors
- Directional/Angle sensors
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# =============================================================================
# FEATURE GROUPING
# =============================================================================

FEATURE_GROUPS = {
    'Temperature': {
        'sensors': [
            'sensor_0_avg',  # Ambient temperature
            'sensor_6_avg',  # Hub controller
            'sensor_7_avg',  # Nacelle controller
            'sensor_9_avg',  # VCP-board
            'sensor_10_avg', # VCS cooling water
            'sensor_11_avg', # Gearbox bearing (high speed)
            'sensor_12_avg', # Gearbox oil
            'sensor_13_avg', # Generator bearing 2
            'sensor_14_avg', # Generator bearing 1
            'sensor_19_avg', # Split ring chamber
            'sensor_21_avg', # IGBT-driver grid side
            'sensor_35_avg', # IGBT-driver rotor side phase1
            'sensor_38_avg', # HV transformer L1
            'sensor_41_avg', # Hydraulic oil
            'sensor_43_avg', # Nacelle temperature
        ],
        'representatives': [
            'sensor_11_avg',  # Gearbox bearing (critical for gearbox faults)
            'sensor_13_avg',  # Generator bearing (critical for generator faults)
            'sensor_41_avg',  # Hydraulic oil (critical for hydraulic faults)
        ],
        'unit': '°C',
        'color': 'Reds'
    },
    
    'Wind': {
        'sensors': [
            'wind_speed_3_avg',
            'wind_speed_3_max',
            'wind_speed_3_min',
            'wind_speed_3_std',
            'wind_speed_4_avg',
        ],
        'representatives': [
            'wind_speed_3_avg',
            'wind_speed_3_std',
        ],
        'unit': 'm/s',
        'color': 'Blues'
    },
    
    'Power': {
        'sensors': [
            'power_29_avg', 'power_29_max', 'power_29_min', 'power_29_std',
            'power_30_avg', 'power_30_max', 'power_30_min', 'power_30_std',
            'reactive_power_27_avg', 'reactive_power_27_max', 'reactive_power_27_min', 'reactive_power_27_std',
            'reactive_power_28_avg', 'reactive_power_28_max', 'reactive_power_28_min', 'reactive_power_28_std',
            'sensor_31_avg', 'sensor_31_max', 'sensor_31_min', 'sensor_31_std',
        ],
        'representatives': [
            'power_29_avg',           # Possible grid active power
            'reactive_power_27_avg',  # Reactive power
        ],
        'unit': 'kW/kVAr',
        'color': 'Greens'
    },
    
    'Electrical': {
        'sensors': [
            'sensor_23_avg',  # Current phase 1
            'sensor_24_avg',  # Current phase 2
            'sensor_25_avg',  # Current phase 3
            'sensor_32_avg',  # Voltage phase 1
            'sensor_33_avg',  # Voltage phase 2
            'sensor_34_avg',  # Voltage phase 3
            'sensor_26_avg',  # Grid frequency
        ],
        'representatives': [
            'sensor_23_avg',  # Current phase 1
            'sensor_32_avg',  # Voltage phase 1
            'sensor_26_avg',  # Grid frequency
        ],
        'unit': 'A/V/Hz',
        'color': 'Purples'
    },
    
    'Mechanical': {
        'sensors': [
            'sensor_18_avg', 'sensor_18_max', 'sensor_18_min', 'sensor_18_std',  # Generator RPM
            'sensor_52_avg', 'sensor_52_max', 'sensor_52_min', 'sensor_52_std',  # Rotor RPM
        ],
        'representatives': [
            'sensor_18_avg',  # Generator RPM
            'sensor_52_avg',  # Rotor RPM
        ],
        'unit': 'rpm',
        'color': 'Oranges'
    },
    
    'Directional': {
        'sensors': [
            'sensor_1_avg',  # Wind absolute direction
            'sensor_2_avg',  # Wind relative direction
            'sensor_5_avg', 'sensor_5_max', 'sensor_5_min', 'sensor_5_std',  # Pitch angle
            'sensor_42_avg', # Nacelle direction
        ],
        'representatives': [
            'sensor_5_avg',   # Pitch angle
            'sensor_42_avg',  # Nacelle direction
        ],
        'unit': 'degrees',
        'color': 'YlOrBr'
    }
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_event_with_labels(event_id, farm_dir, datasets_dir, event_info_df):
    """Load event data with anomaly time information."""
    # Load data
    file_path = os.path.join(datasets_dir, f"{event_id}.csv")
    df = pd.read_csv(file_path, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    
    # Get event info
    event = event_info_df[event_info_df['event_id'] == event_id].iloc[0]
    
    return df, event


def get_window_around_event(df, event, hours_before=72, hours_after=24):
    """
    Extract time window around anomaly event.
    
    Args:
        df: Event dataframe
        event: Event info row
        hours_before: Hours before event_start to include
        hours_after: Hours after event_start to include
    """
    if event['event_label'] == 'normal':
        # For normal events, just take the middle portion
        mid_idx = len(df) // 2
        window_size = int((hours_before + hours_after) * 6)  # 6 timesteps per hour
        start_idx = max(0, mid_idx - window_size // 2)
        end_idx = min(len(df), mid_idx + window_size // 2)
        data = df.iloc[start_idx:end_idx].copy()
        data['hours_to_event'] = 0  # No event
        data['is_anomaly_period'] = False
    else:
        event_start = event['event_start']
        start_time = event_start - timedelta(hours=hours_before)
        end_time = event_start + timedelta(hours=hours_after)
        
        data = df[(df['time_stamp'] >= start_time) & 
                  (df['time_stamp'] <= end_time)].copy()
        
        # Calculate hours to event (negative = before, positive = after)
        data['hours_to_event'] = (data['time_stamp'] - event_start).dt.total_seconds() / 3600
        
        # Mark the 48-hour warning period
        data['is_anomaly_period'] = (data['hours_to_event'] >= -48) & (data['hours_to_event'] <= 0)
    
    return data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_group_trends(group_name, group_config, anomaly_events, normal_events, 
                     farm_dir, datasets_dir, event_info_df, save_dir='results/eda'):
    """
    Plot trends for a sensor group comparing anomaly vs normal events.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    representatives = group_config['representatives']
    n_sensors = len(representatives)
    
    fig, axes = plt.subplots(n_sensors, 2, figsize=(16, 4*n_sensors))
    if n_sensors == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{group_name} Sensors: Anomaly vs Normal Events', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, sensor in enumerate(representatives):
        # Plot anomaly events
        ax_anomaly = axes[idx, 0]
        for event_id in anomaly_events[:3]:  # Plot first 3 anomaly events
            df, event = load_event_with_labels(event_id, farm_dir, datasets_dir, event_info_df)
            if sensor not in df.columns:
                continue
            
            data = get_window_around_event(df, event, hours_before=72, hours_after=12)
            
            if len(data) > 0:
                # Plot sensor value
                ax_anomaly.plot(data['hours_to_event'], data[sensor], 
                              alpha=0.6, linewidth=1.5, 
                              label=f"Event {event_id} ({event.get('event_description', 'Unknown')})")
                
                # Highlight 48-hour warning period
                warning_data = data[data['is_anomaly_period']]
                if len(warning_data) > 0:
                    ax_anomaly.axvspan(warning_data['hours_to_event'].min(),
                                      warning_data['hours_to_event'].max(),
                                      alpha=0.2, color='red', label='48h Warning' if idx == 0 else '')
        
        ax_anomaly.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Fault Start')
        ax_anomaly.set_xlabel('Hours to Fault Event', fontsize=11)
        ax_anomaly.set_ylabel(f'{sensor}\n({group_config["unit"]})', fontsize=10)
        ax_anomaly.set_title(f'Anomaly Events - {sensor}', fontsize=12, fontweight='bold')
        ax_anomaly.legend(fontsize=8, loc='best')
        ax_anomaly.grid(True, alpha=0.3)
        
        # Plot normal events
        ax_normal = axes[idx, 1]
        for event_id in normal_events[:3]:  # Plot first 3 normal events
            df, event = load_event_with_labels(event_id, farm_dir, datasets_dir, event_info_df)
            if sensor not in df.columns:
                continue
                
            data = get_window_around_event(df, event, hours_before=48, hours_after=48)
            
            if len(data) > 0:
                # For normal events, use a different x-axis (just timesteps)
                x_vals = np.arange(len(data)) / 6  # Convert to hours
                ax_normal.plot(x_vals, data[sensor], 
                             alpha=0.6, linewidth=1.5, 
                             label=f"Event {event_id}")
        
        ax_normal.set_xlabel('Time (hours)', fontsize=11)
        ax_normal.set_ylabel(f'{sensor}\n({group_config["unit"]})', fontsize=10)
        ax_normal.set_title(f'Normal Events - {sensor}', fontsize=12, fontweight='bold')
        ax_normal.legend(fontsize=8, loc='best')
        ax_normal.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{group_name.lower()}_trends.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {group_name} trends to: {save_path}")


def plot_aggregated_comparison(group_name, group_config, anomaly_events, normal_events,
                               farm_dir, datasets_dir, event_info_df, save_dir='results/eda'):
    """
    Plot aggregated mean ± std for all sensors in group.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    representatives = group_config['representatives']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'{group_name} Sensors: Aggregated Patterns', 
                 fontsize=14, fontweight='bold')
    
    # Anomaly events
    ax_anomaly = axes[0]
    for sensor in representatives:
        all_curves = []
        
        for event_id in anomaly_events:
            df, event = load_event_with_labels(event_id, farm_dir, datasets_dir, event_info_df)
            if sensor not in df.columns:
                continue
            
            data = get_window_around_event(df, event, hours_before=72, hours_after=12)
            if len(data) > 100:  # Ensure enough data
                # Normalize by subtracting mean
                normalized = data[sensor] - data[sensor].mean()
                all_curves.append(normalized.values)
        
        if all_curves:
            # Interpolate to common length
            min_len = min(len(c) for c in all_curves)
            curves_aligned = [c[:min_len] for c in all_curves]
            
            mean_curve = np.mean(curves_aligned, axis=0)
            std_curve = np.std(curves_aligned, axis=0)
            
            x_vals = np.linspace(-72, 12, min_len)
            ax_anomaly.plot(x_vals, mean_curve, linewidth=2, label=sensor)
            ax_anomaly.fill_between(x_vals, mean_curve - std_curve, mean_curve + std_curve, 
                                   alpha=0.2)
    
    ax_anomaly.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Fault')
    ax_anomaly.axvline(x=-48, color='orange', linestyle=':', linewidth=2, label='48h Before')
    ax_anomaly.set_xlabel('Hours to Fault', fontsize=11)
    ax_anomaly.set_ylabel('Normalized Value (mean-centered)', fontsize=11)
    ax_anomaly.set_title('Anomaly Events (Mean ± Std)', fontsize=12, fontweight='bold')
    ax_anomaly.legend(fontsize=9)
    ax_anomaly.grid(True, alpha=0.3)
    
    # Normal events
    ax_normal = axes[1]
    for sensor in representatives:
        all_curves = []
        
        for event_id in normal_events:
            df, event = load_event_with_labels(event_id, farm_dir, datasets_dir, event_info_df)
            if sensor not in df.columns:
                continue
            
            data = get_window_around_event(df, event, hours_before=48, hours_after=48)
            if len(data) > 100:
                normalized = data[sensor] - data[sensor].mean()
                all_curves.append(normalized.values)
        
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            curves_aligned = [c[:min_len] for c in all_curves]
            
            mean_curve = np.mean(curves_aligned, axis=0)
            std_curve = np.std(curves_aligned, axis=0)
            
            x_vals = np.linspace(0, 96, min_len)
            ax_normal.plot(x_vals, mean_curve, linewidth=2, label=sensor)
            ax_normal.fill_between(x_vals, mean_curve - std_curve, mean_curve + std_curve,
                                 alpha=0.2)
    
    ax_normal.set_xlabel('Time (hours)', fontsize=11)
    ax_normal.set_ylabel('Normalized Value (mean-centered)', fontsize=11)
    ax_normal.set_title('Normal Events (Mean ± Std)', fontsize=12, fontweight='bold')
    ax_normal.legend(fontsize=9)
    ax_normal.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{group_name.lower()}_aggregated.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {group_name} aggregated patterns to: {save_path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS: Sensor Patterns Before Anomalies")
    print("=" * 80)
    
    # Paths
    farm_dir = "Dataset/raw/Wind Farm A"
    datasets_dir = os.path.join(farm_dir, "datasets")
    event_info_path = os.path.join(farm_dir, "event_info.csv")
    
    # Load event info
    event_info_df = pd.read_csv(event_info_path, sep=';')
    event_info_df['event_start'] = pd.to_datetime(event_info_df['event_start'])
    event_info_df['event_end'] = pd.to_datetime(event_info_df['event_end'])
    
    # Separate anomaly and normal events
    anomaly_events = event_info_df[event_info_df['event_label'] == 'anomaly']['event_id'].tolist()
    normal_events = event_info_df[event_info_df['event_label'] == 'normal']['event_id'].tolist()
    
    print(f"\nAnalyzing {len(anomaly_events)} anomaly events and {len(normal_events)} normal events")
    print(f"Feature groups: {list(FEATURE_GROUPS.keys())}")
    
    # Analyze each sensor group
    for group_name, group_config in FEATURE_GROUPS.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {group_name} sensors...")
        print(f"  Representative sensors: {group_config['representatives']}")
        
        # Plot individual trends
        plot_group_trends(group_name, group_config, anomaly_events, normal_events,
                         farm_dir, datasets_dir, event_info_df)
        
        # Plot aggregated patterns
        plot_aggregated_comparison(group_name, group_config, anomaly_events, normal_events,
                                  farm_dir, datasets_dir, event_info_df)
    
    print("\n" + "=" * 80)
    print("✓ EDA Complete! Check results/eda/ directory for visualizations")
    print("=" * 80)


if __name__ == "__main__":
    main()
