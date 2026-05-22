from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "class_imbalance"


def _read_dataset(filename):
    return pd.read_csv(DATA_DIR / filename)


def _scale_features(df, target_column):
    features = df.drop(columns=[target_column])
    target = df[target_column]

    means = features.mean()
    standard_deviations = features.std(ddof=0).replace(0, 1)
    scaled_features = (features - means) / standard_deviations
    return pd.concat([scaled_features, target], axis=1)


def _encode_categoricals(df, categorical_columns):
    return pd.get_dummies(
        df, columns=list(categorical_columns), drop_first=True, dtype=int
    )


def _encode_minority_target(df, target_column):
    df = df.rename(columns={target_column: "target"})
    minority_label = df["target"].value_counts().idxmin()
    df["target"] = (df["target"] == minority_label).astype(int)
    return df


def _load_dataset(
    filename,
    target_column,
    *,
    drop_columns=(),
    categorical_columns=(),
    scale=False,
):
    df = _read_dataset(filename)

    if drop_columns:
        df = df.drop(columns=list(drop_columns))
    if categorical_columns:
        df = _encode_categoricals(df, categorical_columns)
    if scale:
        df = _scale_features(df, target_column)

    return _encode_minority_target(df, target_column)


def df1():
    return _load_dataset("dataset_1039_hiva_agnostic.csv", "label")


def df2():
    return _load_dataset("dataset_312_scene.csv", "FallFoliage")


def df3():
    return _load_dataset("dataset_316_yeast_ml8.csv", "class14")


def df4():
    return _load_dataset("dataset_978_mfeat-factors.csv", "binaryClass", scale=True)


def df5():
    return _load_dataset("dataset_1056_mc1.csv", "c", scale=True)


def df6():
    return _load_dataset(
        "dataset_976_JapaneseVowels.csv",
        "binaryClass",
        drop_columns=("utterance", "frame"),
    )


def df7():
    return _load_dataset("dataset_1020_mfeat-karhunen.csv", "binaryClass", scale=True)


def df8():
    return _load_dataset("dataset_1022_mfeat-pixel.csv", "binaryClass", scale=True)


def df9():
    return _load_dataset("dataset_980_optdigits.csv", "binaryClass", scale=True)


def df10():
    return _load_dataset("dataset_954_spectrometer.csv", "binaryClass", scale=True)


def df11():
    return _load_dataset("dataset_1021_page-blocks.csv", "binaryClass", scale=True)


def df12():
    return _load_dataset("dataset_958_segment.csv", "binaryClass", scale=True)


def df13():
    return _load_dataset(
        "dataset_865_analcatdata_neavote.csv",
        "binaryClass",
        categorical_columns=("Party",),
    )


def df14():
    return _load_dataset(
        "dataset_1013_analcatdata_challenger.csv", "Damaged", scale=True
    )


def df15():
    return _load_dataset(
        "dataset_867_visualizing_livestock.csv",
        "binaryClass",
        categorical_columns=("livestocktype", "country"),
    )


def df16():
    return _load_dataset(
        "dataset_875_analcatdata_chlamydia.csv",
        "binaryClass",
        categorical_columns=("Age", "Gender", "Race"),
    )


def df17():
    return _load_dataset(
        "dataset_764_analcatdata_apnea3.csv",
        "binaryClass",
        categorical_columns=("Automatic", "Scorer_2"),
        scale=True,
    )


def df18():
    # Replacement for the original df18, which was numerically unstable.
    return _load_dataset(
        "dataset_949_arsenic-female-bladder.csv",
        "binaryClass",
        scale=True,
    )


def df19():
    return _load_dataset("dataset_1064_ar6.csv", "defects", scale=True)
