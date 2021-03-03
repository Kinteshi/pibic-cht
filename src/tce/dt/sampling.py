import pandas as pd
import numpy as np


def get_topN_labels_doc(data, label_col, N):
    label_count = data[label_col].value_counts().index.values[0:N]
    slc = [label in label_count for label in data[label_col].values]
    newdata = data[slc]
    return newdata


if __name__ == '__main__':
    pass
