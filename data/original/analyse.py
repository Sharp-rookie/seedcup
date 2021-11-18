import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == "__main__":
    all_info = pd.read_csv("all_info.csv")
    all_info = DataFrame(all_info)
    (all_info.corr("pearson")).to_csv("pearson.csv")
    (all_info.corr("spearman")).to_csv("spearman.csv")
