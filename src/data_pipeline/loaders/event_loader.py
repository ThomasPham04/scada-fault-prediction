"""
EventLoader — data_pipeline.loaders.event_loader
Loads raw event CSV files and event metadata from Wind Farm A.
"""

import os
import pandas as pd


class EventLoader:
    """
    Loads event metadata and per-event SCADA CSV datasets.

    Args:
        farm_dir: Path to the Wind Farm A root directory
                  (contains event_info.csv).
        datasets_dir: Path to the datasets sub-directory
                      (contains <event_id>.csv files).
    """

    def __init__(self, farm_dir: str, datasets_dir: str) -> None:
        self.farm_dir    = farm_dir
        self.datasets_dir = datasets_dir

    def load_event_info(self) -> pd.DataFrame:
        """
        Load event metadata (event_info.csv) from the farm directory.

        Returns:
            DataFrame with columns including event_id, event_label,
            event_start, event_end, asset (turbine id), train_test.
        """
        path = os.path.join(self.farm_dir, "event_info.csv")
        df = pd.read_csv(path, sep=";")
        df["event_start"] = pd.to_datetime(df["event_start"])
        df["event_end"]   = pd.to_datetime(df["event_end"])

        print(f"Loaded event info: {len(df)} events")
        print(f"  - Anomalies: {(df['event_label'] == 'anomaly').sum()}")
        print(f"  - Normal: {(df['event_label'] == 'normal').sum()}")
        return df

    def load_event_data(self, event_id: int) -> pd.DataFrame:
        """
        Load a single event's SCADA CSV dataset by event ID.

        Args:
            event_id: Numeric event identifier.

        Returns:
            DataFrame of 10-minute sensor readings for that event.
        """
        path = os.path.join(self.datasets_dir, f"{event_id}.csv")
        df = pd.read_csv(path, sep=";")
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        return df


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# ---------------------------------------------------------------------------

def load_event_info(farm_dir: str) -> pd.DataFrame:
    """Legacy alias — wraps EventLoader.load_event_info()."""
    loader = EventLoader(farm_dir=farm_dir, datasets_dir="")
    return loader.load_event_info()


def load_event_data(event_id: int, datasets_dir: str) -> pd.DataFrame:
    """Legacy alias — wraps EventLoader.load_event_data()."""
    loader = EventLoader(farm_dir="", datasets_dir=datasets_dir)
    return loader.load_event_data(event_id)
