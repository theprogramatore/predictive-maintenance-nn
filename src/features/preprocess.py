from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "Machine failure"


def split_dataset(df, random_state: int = 42):
    """
    Split the dataset into training, validation, and test sets.

    Final proportions:
    - 70% training
    - 15% validation
    - 15% test

    Stratification preserves the original class distribution
    of Machine failure in all three subsets.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # First split:
    # 70% training
    # 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    # Second split:
    # Half of the temporary set becomes validation
    # and the other half becomes test.
    # This produces 15% validation and 15% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numerical variables are standardized.
    The categorical Type variable is one-hot encoded.

    The preprocessor must be fitted only on training data
    to prevent data leakage.
    """

    numeric = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

def preprocess_splits(X_train, X_val, X_test):
    """
    Fit preprocessing only on the training set and
    transform validation and test sets using the same
    fitted transformations.

    This prevents data leakage.
    """

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor,
    )