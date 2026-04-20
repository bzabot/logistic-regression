from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "class_imbalance"
OUTPUT_DIR = ROOT / "class_imbalance_refactored"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
README_PATH = OUTPUT_DIR / "README.md"

SCENE_LABEL_COLUMNS = [
    "Beach",
    "Sunset",
    "FallFoliage",
    "Field",
    "Mountain",
    "Urban",
]
PREFERRED_POSITIVE_TOKENS = [
    "1",
    "true",
    "t",
    "yes",
    "y",
    "p",
    "positive",
    "pos",
    "sick",
    "def",
]
TARGET_CANDIDATES = [
    "binaryClass",
    "label",
    "Class",
    "class",
    "Damaged",
    "Laid.off",
    "DL",
    "c",
    "defects",
]


def normalize_token(value: object) -> str:
    if pd.isna(value):
        return "__missing__"
    return str(value).strip().lower()


def sanitize_name(value: object) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "column"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def make_unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        base_name = sanitize_name(name)
        suffix = counts.get(base_name, 0)
        unique_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        counts[base_name] = suffix + 1
        unique_names.append(unique_name)

    return unique_names


def choose_positive_label(value_counts: pd.Series) -> object:
    sorted_counts = value_counts.sort_values(ascending=True)
    smallest_count = sorted_counts.iloc[0]
    candidate_labels = [
        label for label, count in sorted_counts.items() if count == smallest_count
    ]
    if len(candidate_labels) == 1:
        return candidate_labels[0]

    normalized_lookup = {
        label: normalize_token(label) for label in candidate_labels
    }
    for token in PREFERRED_POSITIVE_TOKENS:
        for label, normalized in normalized_lookup.items():
            if normalized == token:
                return label

    return sorted(candidate_labels, key=lambda label: str(label))[-1]


def resolve_target_configuration(
    dataset_name: str, columns: list[str]
) -> tuple[str, list[str], str]:
    if dataset_name == "dataset_312_scene.csv":
        return "Urban", SCENE_LABEL_COLUMNS, "selected the Urban label and dropped the other scene labels"

    if dataset_name == "dataset_316_yeast_ml8.csv":
        label_columns = [column for column in columns if column.startswith("class")]
        return (
            "class14",
            label_columns,
            "selected class14 as the target and dropped the other yeast label columns",
        )

    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate, [candidate], ""

    return columns[-1], [columns[-1]], ""


def encode_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    numeric_frame = frame[numeric_columns].copy()
    if not numeric_frame.empty:
        for column in numeric_frame.columns:
            if pd.api.types.is_bool_dtype(numeric_frame[column]):
                numeric_frame[column] = numeric_frame[column].astype(int)
            else:
                numeric_frame[column] = pd.to_numeric(
                    numeric_frame[column], errors="coerce"
                )
        numeric_missing_before = int(numeric_frame.isna().sum().sum())
        fill_values = numeric_frame.mean(numeric_only=True)
        numeric_frame = numeric_frame.fillna(fill_values).fillna(0.0).astype(float)
    else:
        numeric_missing_before = 0

    categorical_frame = frame[categorical_columns].copy()
    if not categorical_frame.empty:
        categorical_missing_before = int(categorical_frame.isna().sum().sum())
        categorical_frame = categorical_frame.astype("string").fillna("__missing__")
        encoded_categorical = pd.get_dummies(
            categorical_frame,
            prefix=categorical_frame.columns,
            prefix_sep="__",
            drop_first=True,
            dtype=float,
        )
    else:
        categorical_missing_before = 0
        encoded_categorical = pd.DataFrame(index=frame.index)

    encoded = pd.concat([numeric_frame, encoded_categorical], axis=1)
    constant_columns = [
        column for column in encoded.columns if encoded[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        encoded = encoded.drop(columns=constant_columns)

    encoded.columns = make_unique_names(list(encoded.columns))
    encoded = encoded.reset_index(drop=True)

    metadata = {
        "numeric_feature_columns": len(numeric_columns),
        "categorical_feature_columns": len(categorical_columns),
        "numeric_missing_values_imputed": numeric_missing_before,
        "categorical_missing_values_filled": categorical_missing_before,
        "constant_feature_columns_dropped": len(constant_columns),
        "encoded_feature_columns": encoded.shape[1],
    }
    return encoded, metadata


def write_readme() -> None:
    content = """# Refactored Class Imbalance Datasets

Each CSV in this folder mirrors one source file from `class_imbalance/`, but in a format that is easier to feed into the NumPy logistic regression baseline:

- every output file keeps the original dataset filename
- all feature columns are numeric
- the label column is always named `target`
- `target = 1` is the minority class for that dataset
- known multi-label files have their extra label columns removed from the features

These files are convenient for loading and benchmarking, but they are not a substitute for split-aware preprocessing. If you want leakage-safe evaluation, do train/test splitting first and then fit scaling on the training split only.
"""
    README_PATH.write_text(content, encoding="utf-8")


def process_dataset(path: Path) -> dict[str, object]:
    source = pd.read_csv(path)
    rows_in, columns_in = source.shape
    target_column, related_label_columns, note = resolve_target_configuration(
        path.name, list(source.columns)
    )
    auxiliary_label_columns = [
        column for column in related_label_columns if column != target_column
    ]

    dataset = source.copy()
    missing_target_rows = int(dataset[target_column].isna().sum())
    if missing_target_rows:
        dataset = dataset.loc[dataset[target_column].notna()].copy()

    target_source = dataset[target_column]
    target_counts = target_source.value_counts(dropna=False)
    if len(target_counts) != 2:
        raise ValueError(
            f"{path.name} does not have exactly two target classes in {target_column!r}"
        )

    positive_label = choose_positive_label(target_counts)
    negative_label = next(label for label in target_counts.index if label != positive_label)
    target = target_source.map({negative_label: 0, positive_label: 1}).astype(int)

    feature_frame = dataset.drop(columns=auxiliary_label_columns + [target_column])
    encoded_features, feature_metadata = encode_features(feature_frame)
    encoded_features = pd.concat(
        [encoded_features, target.reset_index(drop=True).rename("target")], axis=1
    )
    encoded_features.to_csv(OUTPUT_DIR / path.name, index=False)

    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())
    mapping = json.dumps(
        {str(negative_label): 0, str(positive_label): 1}, ensure_ascii=True
    )

    return {
        "source_file": path.name,
        "output_file": path.name,
        "rows_in": rows_in,
        "rows_out": encoded_features.shape[0],
        "columns_in": columns_in,
        "feature_columns_out": encoded_features.shape[1] - 1,
        "target_column_source": target_column,
        "target_mapping": mapping,
        "negative_label_source": str(negative_label),
        "positive_label_source": str(positive_label),
        "negative_count": negative_count,
        "positive_count": positive_count,
        "auxiliary_label_columns_dropped": len(auxiliary_label_columns),
        "auxiliary_label_column_names": ",".join(auxiliary_label_columns),
        "missing_target_rows_dropped": missing_target_rows,
        "note": note,
        **feature_metadata,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest_rows = [
        process_dataset(path) for path in sorted(SOURCE_DIR.glob("*.csv"))
    ]
    manifest = pd.DataFrame(manifest_rows).sort_values("source_file")
    manifest.to_csv(MANIFEST_PATH, index=False)
    write_readme()
    print(f"Refactored {len(manifest)} datasets into {OUTPUT_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
