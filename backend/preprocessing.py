import pandas as pd
from . import model_loader

def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    meta = model_loader.preprocessing_meta
    
    # Validate against reconstructed raw features
    expected_cols = set(model_loader.raw_feature_columns)
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required input columns: {', '.join(missing_cols)}")
    
    # 1. Drop id, label and redundant columns
    cols_to_drop = []
    if meta.get("id_column_dropped") in df.columns:
        cols_to_drop.append(meta["id_column_dropped"])
    if meta.get("label_column_dropped") in df.columns:
        cols_to_drop.append(meta["label_column_dropped"])
        
    for col in meta.get("redundant_columns_dropped", []):
        if col in df.columns:
            cols_to_drop.append(col)
            
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. One-hot encode
    cat_cols = [col for col in meta.get("categorical_columns", []) if col in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=meta.get("onehot_drop_first", True))
    
    # 3. Reindex to match encoded_column_order exactly
    encoded_order = meta.get("encoded_column_order", [])
    
    # Missing columns filled with 0 (since they are likely missing dummy columns)
    for col in encoded_order:
        if col not in df.columns:
            df[col] = 0
            
    # Drop unexpected columns and ensure order
    df = df[encoded_order]
    
    # 4. Return aligned, numeric-only dataframe
    return df
