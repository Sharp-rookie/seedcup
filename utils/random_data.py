import random

import pandas as pd


def generate_new_data(path="../data/original/all_info.csv"):
    df = pd.DataFrame(pd.read_csv(path))
    column = list(df.columns)
    df = list(df.values)
    random.shuffle(df)
    train_df = pd.DataFrame(columns=column, data=df[:int(len(df)*4/5)]).fillna(0)
    valid_df = pd.DataFrame(columns=column, data=df[int(len(df)*4/5):]).fillna(0)
    train_df.to_csv(index=False, path_or_buf="../data/random_data/train.csv")
    valid_df.to_csv(index=False, path_or_buf="../data/random_data/valid.csv")


if __name__ == "__main__":
    generate_new_data()
