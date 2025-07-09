import pandas as pd

def vote(series):
    modes = series.mode()
    if not modes.empty: # 检查 modes Series 是否不为空
        return modes.iloc[0] # 返回第一个众数
    else:
        return series.max() # 如果没有众数，返回该 series 的最大值

def vote_true(series):
    # 真实按照最大的选
    modes = series.mode()
    if not modes.empty: # 检查 modes Series 是否不为空
        return modes.max()  # 改为返回最大众数
    else:
        return series.max() # 如果没有众数，返回该 series 的最大值

def vote_pen(series): 
    if 3 in series.values:
        return 3  # 如果包含2，直接返回2
    modes = series.mode()
    if not modes.empty:
        return modes.iloc[0]  # 否则返回众数
    else:
        return series.max()  # 如果没有众数，返回最大值

def vote_pen_v2(series):
    if 3 in series.values:
        return 3  # 如果包含3，直接返回3
    if 2 in series.values:
        return 2  # 如果包含2，直接返回2
    modes = series.mode()
    if not modes.empty:
        return modes.iloc[0]  # 否则返回众数
    else:
        return series.max()  # 如果没有众数，返回最大值

def get_id(x):
    return x.split("_")[0]

def prediction_refine_vote(df,col_name):
    df['ID_t'] = df.index.map(get_id)
    df[col_name] = df.groupby('ID_t')[col_name].transform(vote)
    return df.drop(columns=['ID_t'])

def prediction_refine_vote_true(df,col_name):
    df['ID_t'] = df.index.map(get_id)
    df[col_name] = df.groupby('ID_t')[col_name].transform(vote_true)
    return df.drop(columns=['ID_t'])

def prediction_refine_vote_pen(df,col_name):
    df['ID_t'] = df.index.map(get_id)
    df[col_name] = df.groupby('ID_t')[col_name].transform(vote_pen)
    return df.drop(columns=['ID_t'])

def prediction_refine_vote_pen_v2(df,col_name):
    df['ID_t'] = df.index.map(get_id)
    df[col_name] = df.groupby('ID_t')[col_name].transform(vote_pen_v2)
    return df.drop(columns=['ID_t'])