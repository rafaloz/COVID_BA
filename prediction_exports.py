"""Model-specific prediction exports; importing this module performs no analysis."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd


def corrected_columns(frame):
    return [column for column in frame.columns
            if any(column == stem or column.startswith(stem + "_")
                   for stem in ("pred_Edad_c", "BrainPAD_c"))]


def update_prededad(results_df, predictions_df, id_col_results="ID",
                    id_col_pred="id", pred_col="prediction"):
    """Attach raw pyment predictions without retaining another model's correction.

    Inputs are copied. IDs represented as integer-valued CSV floats are
    normalised on both sides; missing or conflicting predictions fail explicitly.
    """
    result = results_df.drop(columns=corrected_columns(results_df)).copy()
    predictions = predictions_df[[id_col_pred, pred_col]].copy()

    def ids(values):
        return values.astype(str).str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)

    result[id_col_results] = ids(result[id_col_results])
    predictions[id_col_pred] = ids(predictions[id_col_pred])
    conflicts = predictions.groupby(id_col_pred)[pred_col].nunique(dropna=False)
    if conflicts.gt(1).any():
        raise ValueError("Conflicting pyment predictions for duplicate IDs")
    predictions = predictions.drop_duplicates(id_col_pred, keep="last")
    result["pred_Edad"] = result[id_col_results].map(
        predictions.set_index(id_col_pred)[pred_col])
    if result["pred_Edad"].isna().any():
        raise ValueError("Missing pyment predictions for one or more participants")
    if "Edad" in result:
        result["BrainPAD"] = result["pred_Edad"] - result["Edad"]
    return result


class PredictionExporter:
    """Write one model/run per directory and refuse to overwrite an export."""
    def __init__(self, model, output_root):
        if model not in {"morphometric_harmonized", "morphometric_unharmonized", "pyment"}:
            raise ValueError("Unknown prediction model")
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        self.model = model
        self.directory = Path(output_root) / model / run
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = {}

    def write(self, frame, filename):
        if Path(filename).name != filename or not filename.endswith(".csv"):
            raise ValueError("Export name must be a CSV filename without a directory")
        if self.model == "pyment" and corrected_columns(frame):
            raise ValueError("Pyment export contains inherited corrected columns")
        destination = self.directory / filename
        frame.to_csv(destination, index=False, mode="x")
        self.files[filename] = {
            "rows": len(frame), "columns": list(frame.columns),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        manifest = {"model": self.model, "files": self.files}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return destination
