"""
Event Loader — data_pipeline.loaders.event_loader
Loads raw event CSV files and event metadata from Wind Farm A.
"""

import os
import pandas as pd


def load_event_info(farm_dir: str) -> pd.DataFrame:
    """Load event metadata (event_info.csv) from the farm directory."""
    event_info_path = os.path.join(farm_dir, "event_info.csv")
    df = pd.read_csv(event_info_path, sep=';')
    df['event_start'] = pd.to_datetime(df['event_start'])
    df['event_end'] = pd.to_datetime(df['event_end'])

    print(f"Loaded event info: {len(df)} events")
    print(f"  - Anomalies: {(df['event_label'] == 'anomaly').sum()}")
    print(f"  - Normal: {(df['event_label'] == 'normal').sum()}")

    return df


def load_event_data(event_id: int, datasets_dir: str) -> pd.DataFrame:
    """Load a single event's SCADA CSV dataset by event ID."""
    file_path = os.path.join(datasets_dir, f"{event_id}.csv")
    df = pd.read_csv(file_path, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df
