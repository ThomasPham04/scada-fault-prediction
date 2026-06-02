"""
GroundTruth - data_pipeline.preprocessing.ground_truth

Creates row-level ground truth labels for CARE event files.

Label definition:
    0 -> Normal
    1 -> Fault/anomaly row

A row is positive when it has a non-zero CARE status code, or when it belongs
to the prediction segment of an event whose event-level label is anomaly.
"""

import pandas as pd


class GroundTruth:
    """
    Produces row-level ground truth labels for CARE to Compare event datasets.

    Args:
        event_info: DataFrame from EventLoader.load_event_info().
                    Must contain: event_id, event_label.
    """

    def __init__(self, event_info: pd.DataFrame) -> None:
        required = {"event_id", "event_label"}
        missing = required - set(event_info.columns)
        if missing:
            raise ValueError(f"event_info is missing columns: {missing}")
        self._info = event_info.set_index("event_id")

    # ------------------------------------------------------------------
    # Core label creation
    # ------------------------------------------------------------------

    def make_labels(self, df: pd.DataFrame, event_id: int) -> pd.Series:
        """
        Integer labels (0 = normal, 1 = fault) for a single event dataset.

        A row receives label=1 when status_type_id is non-zero, or when it is
        in the prediction segment of an anomaly event.

        Args:
            df:       Full event DataFrame - must have status_type_id and train_test.
            event_id: Numeric event identifier matching event_info.

        Returns:
            pd.Series[int] with the same index as df, named 'label'.
        """
        event = self._info.loc[event_id]
        labels = pd.Series(0, index=df.index, dtype=int, name="label")

        fault_mask = df["status_type_id"] != 0
        if event["event_label"] == "anomaly":
            fault_mask = fault_mask | (df["train_test"] == "prediction")
        labels[fault_mask] = 1

        return labels

    def make_normal_index(self, df: pd.DataFrame, event_id: int) -> pd.Series:
        """
        Boolean normal_index matching the authors' convention.

        True  = normal row  (label == 0)
        False = fault row   (label == 1)

        This mirrors care2compare.py's normal_index convention for rows that
        are considered normal.

        Args:
            df:       Full event DataFrame - must have status_type_id and train_test.
            event_id: Numeric event identifier.

        Returns:
            pd.Series[bool] - True means normal, False means fault.
        """
        return self.make_labels(df, event_id) == 0

    # ------------------------------------------------------------------
    # Convenience: add label column directly to a DataFrame copy
    # ------------------------------------------------------------------

    def add_label_column(self, df: pd.DataFrame, event_id: int) -> pd.DataFrame:
        """
        Return a copy of df with a 'label' column added.

        Args:
            df:       Event DataFrame - must have status_type_id and train_test.
            event_id: Numeric event identifier.

        Returns:
            DataFrame copy with 'label' column appended.
        """
        out = df.copy()
        out["label"] = self.make_labels(df, event_id).values
        return out

    # ------------------------------------------------------------------
    # Convenience: label summary
    # ------------------------------------------------------------------

    def summary(self, df: pd.DataFrame, event_id: int) -> None:
        """Print label distribution for an event dataset."""
        event = self._info.loc[event_id]
        labels = self.make_labels(df, event_id)
        n_fault = labels.sum()
        n_normal = len(labels) - n_fault
        print(
            f"Event {event_id} ({event['event_label']:7s}) | "
            f"LabelRule=status_type_id!=0 OR anomaly prediction segment | "
            f"Normal={n_normal:,}  Fault={n_fault:,}  Total={len(labels):,}"
        )
