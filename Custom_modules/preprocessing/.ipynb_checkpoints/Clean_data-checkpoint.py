# preprocessing/clean_data.py

import pandas as pd

# =====================================================
# Normalize Columns
# =====================================================


def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "pct")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


# =====================================================
# Capping Outliers
# =====================================================
def cap_outliers_iqr(df):
    """
    Detects and caps outliers using IQR to prevent centroid skewing in KMeans.
    """
    df_clean = df.copy()
    for col in df.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping outliers
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clean


# =====================================================
# 1. REMOVE DUPLICATES
# =====================================================


def remove_duplicates(df):
    """
    Removes duplicate rows from dataset.
    """

    initial_rows = len(df)

    df = df.drop_duplicates()

    final_rows = len(df)

    print(f"\n✅ Removed {initial_rows - final_rows} duplicate rows")

    return df


# =====================================================
# 2. CLEAN TEXT COLUMNS
# =====================================================


def clean_text_columns(df, lowercase_cols=None):
    """
    Cleans object/string columns.

    Operations:
    - removes leading/trailing spaces
    - removes multiple spaces
    - optional lowercase conversion
    """

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:

        # Convert to string
        df[col] = df[col].astype(str)

        # Remove leading/trailing spaces
        df[col] = df[col].str.strip()

        # Remove multiple spaces
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # ---------------------------------------------
    # OPTIONAL LOWERCASE CONVERSION
    # ---------------------------------------------

    if lowercase_cols:

        for col in lowercase_cols:

            if col in df.columns:
                df[col] = df[col].str.lower()

    print("\n✅ Text columns cleaned")

    return df


# =====================================================
# 3. CONVERT DATE COLUMNS
# =====================================================


def convert_date_columns(df, date_cols):
    """
    Converts specified columns to datetime.
    """

    for col in date_cols:

        if col in df.columns:

            df[col] = pd.to_datetime(df[col], errors="coerce")

            print(f"✅ Converted {col} to datetime")

        else:
            print(f"⚠ Column '{col}' not found")

    return df


# =====================================================
# 4. HANDLE MISSING VALUES
# =====================================================


def handle_missing_values(df, numeric_strategy="median", categorical_strategy="mode"):
    """
    Handles missing values separately for:
    - numeric columns
    - categorical columns
    """

    # ---------------------------------------------
    # NUMERIC COLUMNS
    # ---------------------------------------------

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:

        if df[col].isnull().sum() > 0:

            if numeric_strategy == "mean":

                df[col] = df[col].fillna(df[col].mean())

            elif numeric_strategy == "median":

                df[col] = df[col].fillna(df[col].median())

    # ---------------------------------------------
    # CATEGORICAL COLUMNS
    # ---------------------------------------------

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:

        if df[col].isnull().sum() > 0:

            if categorical_strategy == "mode":

                df[col] = df[col].fillna(df[col].mode()[0])

            elif categorical_strategy == "unknown":

                df[col] = df[col].fillna("Unknown")

    print("\n✅ Missing values handled")

    return df


# =====================================================
# 5. MASTER CLEANING PIPELINE
# =====================================================


def clean_dataset(
    df,
    date_cols=None,
    lowercase_cols=None,
    numeric_strategy="median",
    categorical_strategy="mode",
):
    """
    Complete data cleaning pipeline.
    """

    print("\n" + "=" * 50)
    print("STARTING DATA CLEANING")
    print("=" * 50)

    # ---------------------------------------------
    # STEP 1
    # ---------------------------------------------

    df = remove_duplicates(df)

    # ---------------------------------------------
    # STEP 2
    # ---------------------------------------------

    df = clean_text_columns(df, lowercase_cols=lowercase_cols)

    # ---------------------------------------------
    # STEP 3
    # ---------------------------------------------

    if date_cols:

        df = convert_date_columns(df, date_cols=date_cols)

    # ---------------------------------------------
    # STEP 4
    # ---------------------------------------------

    df = handle_missing_values(
        df, numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy
    )

    # ---------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------

    print("\n✅ Dataset Cleaning Completed")

    print("\nFinal Dataset Shape:", df.shape)

    print("\nData Types:\n")
    print(df.dtypes)

    return df
