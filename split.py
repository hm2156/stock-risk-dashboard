import pandas as pd

def time_split(df: pd.DataFrame):
    n      = len(df)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    return train, val, test
