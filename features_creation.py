import pandas as pd
import numpy as np
import pickle
from os import makedirs
from typing import Callable

def window_sliding(data: pd.DataFrame, features: dict[str, list[tuple]], in_place=False):
    if not in_place:
        data = data.copy()
    
    for feature, functions in features.items():
        for func, sizes in functions.items():
            for w_size in sizes:
                rolling = data[feature].rolling(window=w_size, min_periods=1)
                data[f"{feature}_{func.__name__}_{w_size}h"] = rolling.agg(func)
    
    if not in_place:
        return data

    
def save_data(folder_name: str, features_scaled, Y, data):
    makedirs(f'data/{folder_name}')
    data.to_csv(f'data/{folder_name}/data.csv', index=False)
    Y.to_csv(f'data/{folder_name}/y.csv', index=False)
    pickle.dump(features_scaled, open(f'data/{folder_name}/features_scaled', 'wb'))  