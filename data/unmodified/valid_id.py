import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

if __name__ == "__main__":
    t = DataFrame(pd.read_csv("valid.csv"))
    l = np.array(t['id']).tolist()
    with open("valid_id.txt", "w") as f:
        f.write(str(l))
