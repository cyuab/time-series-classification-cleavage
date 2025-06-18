import pandas as pd
import numpy as np
from ast import literal_eval

def load_as_pd_multiindex(file_path):
    """
    Load a file as pd_multiindex.
    https://www.sktime.net/en/stable/examples/AA_datatypes_and_datasets.html#Section-1.2.1:-Time-series-panels---the-%22pd-multiindex%22-mtype
    - file_path: E.g., "../data/01_single.csv"
    """
    df = pd.read_csv(file_path, low_memory=False)
    # df.iloc[0].five_p_cleav_1[0]
    # Some problem in reading the data.
    # https://stackoverflow.com/questions/79413934/write-read-columns-of-list-of-numbers-integer-or-float-to-from-csv-in-python
    # https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list/63020659#63020659
    df = df.map(literal_eval)
    # print(df.iloc[0].size, df.shape[0], len(df), df.iloc[0].iloc[0])
    # 8 827 827 [-2, 2, -2, 2, 1, -2, -2, -2, -2, 2, 1, 1, 1, -2]
    # 8: no of columns (features) an instance has
    # 827: no of instances
    # [-2, 2, -2, 2, 1, -2, -2, -2, -2, 2, 1, 1, 1, -2]: first instance's first feature (i.e., five_p_cleav_1)
    #
    for i in range(len(df)):
        for j in range(df.iloc[0].size):
            # https://stackoverflow.com/questions/19482970/get-a-list-from-pandas-dataframe-column-headers
            col = pd.DataFrame(df.iloc[i].iloc[j], columns=[df.columns.values[j]]) # E.g., five_p_cleav_1
            if j == 0:
                col_all = col
            else:
                # 【pandas数据合并一】：pd.concat()用法
                # https://blog.csdn.net/xue_11/article/details/118424380  
                col_all = pd.concat([col_all, col], axis=1)
        # https://stackoverflow.com/questions/25457920/convert-row-names-into-a-column-in-pandas
        col_all.index.name = 'time_points'
        col_all.reset_index(inplace=True)
        # https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
        col_all.insert(0, 'instances', i)
        if i == 0:
            rows = col_all
        else:
            rows = pd.concat([rows, col_all], axis=0)
    rows = rows.set_index(["instances", "time_points"])
    return rows

def construct_X(ts_panel, include_five_p_cleav, include_prob, include_ss):
    if include_five_p_cleav:
        prefix = "five_p"
    else:
        prefix = "three_p"
        # pos_instances
    pos_instances = ts_panel[[prefix + "_cleav_1", prefix + "_cleav_compl_1"]]
    pos_instances_len = len(pos_instances.index.get_level_values(0).unique())
    # neg_instances
    neg_instances = ts_panel[[prefix + "_non_cleav_1", prefix + "_non_cleav_compl_1"]]
    neg_instances = neg_instances.rename(columns={prefix + '_non_cleav_1': prefix + '_cleav_1', prefix + '_non_cleav_compl_1': prefix + '_cleav_compl_1'})
    if prefix + '_cleav_2' in ts_panel.columns:
        # pos_instances
        pos_instances =  pd.concat([pos_instances, ts_panel[[prefix+"_cleav_2", prefix+"_cleav_compl_2"]]], axis=1)
        # neg_instances
        neg_instances =  pd.concat([neg_instances, ts_panel[[prefix+"_non_cleav_2", prefix+"_non_cleav_compl_2"]]], axis=1)
        neg_instances = neg_instances.rename(columns={prefix+'_non_cleav_2': prefix+'_cleav_2', prefix+'_non_cleav_compl_2': prefix+'_cleav_compl_2'})
    # https://stackoverflow.com/questions/79445936/shift-change-the-index-of-a-dataframe
    if include_prob:
        pos_instances =  pd.concat([pos_instances, ts_panel[[prefix+"_cleav_prob"]]], axis=1)
        # neg_instances
        neg_instances =  pd.concat([neg_instances, ts_panel[[prefix+"_non_cleav_prob"]]], axis=1)
        neg_instances = neg_instances.rename(columns={prefix+"_non_cleav_prob": prefix+"_cleav_prob"})
    
    if include_ss:
        pos_instances =  pd.concat([pos_instances, ts_panel[[prefix+"_cleav_ss"]]], axis=1)
        # neg_instances
        neg_instances =  pd.concat([neg_instances, ts_panel[[prefix+"_non_cleav_ss"]]], axis=1)
        neg_instances = neg_instances.rename(columns={prefix+"_non_cleav_ss": prefix+"_cleav_ss"})
    neg_instances.index = neg_instances.index.map(lambda idx: (idx[0] + pos_instances_len, idx[1])) 
    X = pd.concat([pos_instances, neg_instances], axis=0)
    return X  

def pad_multiindex_ts(df, pad_value=0):
    """
    Pads each time series (per instance) in a MultiIndex DataFrame to equal length.
    
    Parameters:
        df: pd.DataFrame with MultiIndex (instance, time)
        pad_value: value to use for padding

    Returns:
        padded_df: pd.DataFrame with same MultiIndex structure, all time series equal length
    """
    instances = df.index.get_level_values(0).unique()
    time_lengths = df.groupby(level=0).size()
    max_len = time_lengths.max()

    padded_dfs = []

    for instance in instances:
        ts = df.loc[instance]
        pad_len = max_len - len(ts)
        if pad_len > 0:
            pad_df = pd.DataFrame(
                pad_value,
                index=pd.RangeIndex(len(ts), len(ts) + pad_len),
                columns=ts.columns
            )
            ts = pd.concat([ts, pad_df])
        # Rebuild MultiIndex
        ts.index = pd.MultiIndex.from_product([[instance], ts.index])
        padded_dfs.append(ts)

    return pd.concat(padded_dfs)