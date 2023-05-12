"""this code is written by Sujith"""

import enum
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import msgpack
import msgpack_numpy as mpn
from scipy.interpolate import interp1d
from more_itertools import locate

def read_df_csv(filename, offset=2):
    """
    this function reads the csv file from motion capture system
    and makes it into a dataframe and returns only useful information

    filename: input path to csv file from motive
    offset:to ignore the first two columns with time and frames generally

    """

    # offset = 2 #first two columns with frame_no and time

    pth = filename
    raw = pd.read_csv(pth)
    cols_list = raw.columns     # first row which contains capture start time
    inx = [i for i, x in enumerate(cols_list) if x == "Capture Start Time"]
    st_time = cols_list[inx[0] + 1]
    st_time = datetime.strptime(st_time, "%Y-%m-%d %I.%M.%S.%f %p")  # returns datetime object

    mr_inx = pd.read_csv(pth, skiprows=3)
    markers_raw = mr_inx.columns
    marker_offset = offset  # for ignoring time and frame cols
    markers_raw = markers_raw[marker_offset:]
    col_names = []
    for i in range(0, len(markers_raw), 3):
        col_names.append(markers_raw[i].split(":")[1])

    df_headers = ["frame", "seconds"]

    for id, i in enumerate(col_names):
        if not i.islower():
            col_names[id] = i.lower()
    
    for i in col_names:
        df_headers.append(i + "_x")
        df_headers.append(i + "_y")
        df_headers.append(i + "_z")
    mo_data = pd.read_csv(pth, skiprows=6)
    # mo_data = mo_data.rename(mo_data.columns, df_headers)
    mo_data.columns = df_headers

    return mo_data, st_time


def add_datetime_col(df, _time, _name):

    """
    df:     dataframe
    _time:  the time you want to start your column with
    _name:  name of the column that has time in seconds
    """

    _t = []
    for i in list(df[_name]):
        _t.append(_time + timedelta(0,float(i)))
    df["time"] = _t
    return df
    
def add_datetime_diff(df, _time, _sync, _diff_name, truncate = False):

    """
    df:         dataframe
    _time:      the time you want to start your column with
    _sync:      Name of the column with external sync which turns 1 when your recording 
                your data and 0 when your not
    _diff_name: external time/ other time which you want to find the difference 
                and add them to the _time column to sync clock
    truncate:   truncate will also cut the values after your _sync goes to 0
    """
    _inx = 0
    for inx, i in enumerate(df[_sync]):
        if i == 1:
            _inx = inx
            break

    df = df.loc[_inx:].copy()     # dropping unnecessary rows

    _diff = list(df[_diff_name].diff())
    _sum = 0
    _t = []
    for i in _diff:
        if np.isnan(i):
            i = 0
        _sum = _sum + i
        _t.append(_time + timedelta(0,float(_sum/1000)))

    df["time"] = _t

    if truncate:
        _count = 0
        for count, j in enumerate(df[_sync]):
            if j == 0:
                _count = count
                print(_count)
                break
        if _count is not 0:   
            df = df.loc[:_count].copy()     # dropping unnecessary rows
    
    return df

def add_time_from_file(df, _pth):
    """
    df:     dataframe
    _pth:   path to the file which has time in seconds
    """
    with open(_pth, "rb") as f:
        _time_obj = msgpack.Unpacker(f)
        _time = []
        for i in _time_obj:
            _time.append(i)
    df["time"] = _time
    df["time"] = pd.to_datetime(df["time"])
    return df

def interpolate_target_df(target_df, reference_df, col_names = None):
    # based on mocap time using interp1d function
    # for x, y, z coordinates and yaw, pitch, roll
    """
    target_df: dataframe to be interpolated
    reference_df: dataframe to be used as reference
    col_names: list of column names to be interpolated

    both dataframes should have a column named "time"
    """

    target_df = target_df.reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)

    # column names 
    if col_names is None:
        col_names = ["frame_id","x", "y", "z", "yaw", "pitch", "roll"]

    df = pd.DataFrame(columns=col_names)
    df["time"] = reference_df.time

    # change reference time to float
    reference_df["time"] = reference_df["time"].dt.hour * 3600 + reference_df["time"].dt.minute * 60 + reference_df["time"].dt.second + reference_df["time"].dt.microsecond / 1000000
    # change target time to float
    target_df["time"] = target_df["time"].dt.hour * 3600 + target_df["time"].dt.minute * 60 + target_df["time"].dt.second + target_df["time"].dt.microsecond / 1000000

    new_cols = []
    for i in col_names:
        f = interp1d(target_df.time, target_df[i], fill_value="extrapolate")

        _new_val = f(reference_df.time)

        new_cols.append(_new_val)

    for idx, i in enumerate(col_names):
        df[i] = new_cols[idx]
    
    
    return df

def trunkate_dfs(df_1, df_2, display_print = False):
    # which starts earlier
    """
    df_1: dataframe 1
    df_2: dataframe 2

    both dataframes should have a column named "time"

    this code will truncate the dataframes to the same "time" length
    """
    if df_1["time"].iloc[0] < df_2["time"].iloc[0]:
        _start_inx = df_1.time.searchsorted(df_2.time[0])

        df_1 = df_1.loc[_start_inx:]
        df_1 = df_1.reset_index(drop=True)
        if display_print:
            print(_start_inx)
            print("df_1 starts earlier")
    else:
        _start_inx = df_2.time.searchsorted(df_1.time[0])
        df_2 = df_2.loc[_start_inx:]
        df_2 = df_2.reset_index(drop=True)
        if display_print:
            print(_start_inx)
            print("df_2 starts earlier")

    # which ends later
    if df_1["time"].iloc[-1] > df_2["time"].iloc[-1]:
        _end_inx = df_1.time.searchsorted(df_2.time.iloc[-1])
        df_1 = df_1.loc[:_end_inx]
        df_1 = df_1.reset_index(drop=True)
        if display_print:
            print(_end_inx)
            print("df_1 ends later")
    else:
        _end_inx = df_2.time.searchsorted(df_1.time.iloc[-1])
        df_2 = df_2.loc[:_end_inx]
        df_2 = df_2.reset_index(drop=True)
        if display_print:
            print(_end_inx)
            print("df_2 ends later")

    return df_1, df_2

def read_rigid_body_csv(_pth):
    """
    _pth: path to the rigid body file
    """
    df = pd.read_csv(_pth, skiprows=2, header=None, dtype=str)

    # get the start time of the capture
    raw_df = pd.read_csv(_pth, dtype=str)
    cols_list = raw_df.columns     # first row which contains capture start time
    inx = [i for i, x in enumerate(cols_list) if x == "Capture Start Time"]
    st_time = cols_list[inx[0] + 1]
    st_time = datetime.strptime(st_time, "%Y-%m-%d %I.%M.%S.%f %p")  # returns datetime object

    _marker_type = []
    for idx in df.columns:
        _marker_type.append(df[idx][0])

    def find_indices(list_to_check, item_to_find):
        indices = locate(list_to_check, lambda x: x == item_to_find)
        return list(indices)

    # find the indices of the marker types
    _rb_idx = find_indices(_marker_type, "Rigid Body")
    _rb_marker_idx = find_indices(_marker_type, "Rigid Body Marker")
    _marker_idx = find_indices(_marker_type, "Marker")

    _rb_idx.sort()
    _rb_marker_idx.sort()
    _marker_idx.sort()

    _rb_df = df.copy(deep=True)
    _rb_df = df[1:]
    _analysis_type = list(_rb_df.iloc[2].values)

    _rotation_ids = find_indices(_analysis_type, "Rotation")
    _rotation_ids.sort()

    _rb_pos_idx = _rb_idx.copy()

    # remove the rotation indices from the position indices
    [_rb_pos_idx.remove(i) for i in _rotation_ids]

    # create column names for angle
    col_names = []

    # first two columns
    col_names.append("frame")
    col_names.append("seconds")

    for i in _rotation_ids:
        _col = _rb_df[i].iloc[3].lower()
        col_names.append("rb_ang_" + _col)

    # create column names for position
    for i in _rb_pos_idx:
        if isinstance(_rb_df[i].iloc[3], str):
            _col = _rb_df[i].iloc[3].lower()
            col_names.append("rb_pos_" + _col)
        else:
            col_names.append("rb_pos_err")


    # rigid body individual marker section

    for i in _rb_marker_idx:
        if isinstance(_rb_df[i].iloc[0], str):
            _col_head = _rb_df[i].iloc[0].lower()
            _col_head = _col_head.split(":")[1].strip()
            _col_head = _col_head.replace("marker", "")
            _m_idx = int(_col_head)
            
            if isinstance(_rb_df[i].iloc[3], str):
                _col = _rb_df[i].iloc[3].lower()
                col_names.append("rb_marker_m" + str(_col_head) + "_" + _col)
            else:
                col_names.append("rb_marker_m" + str(_col_head) + "_mq") # marker quality


    # individual marker section 

    for i in _marker_idx:
        if isinstance(_rb_df[i].iloc[0], str):
            _col_head = _rb_df[i].iloc[0].lower()
            _col_head = _col_head.split(":")[1].strip()
            _col_head = _col_head.replace("marker", "")
            _m_idx = int(_col_head)
            
            if isinstance(_rb_df[i].iloc[3], str):
                _col = _rb_df[i].iloc[3].lower()
                col_names.append("m" + str(_col_head) + "_" + _col)

    _rb_df = _rb_df[4:]
    _rb_df.columns = col_names

    #reset index
    _rb_df = _rb_df.reset_index(drop=True)
    _rb_df = _rb_df.apply(pd.to_numeric, errors='ignore')


    return _rb_df, st_time


def get_marker_name(val):
    # int value to marker name
    """
    this function is for rigid body markers
    it generates the marker names for the rigid body markers
    using the given integer value    
    """
    # create using dictionary
    _val = {"x":"m" + str(val) + "_x",
            "y":"m" + str(val) + "_y",
            "z":"m" + str(val) + "_z",}

    return _val

def get_rb_marker_name(val):
    # int value to marker name
    """
    this function is for rigid body markers
    it generates the marker names for the rigid body markers
    using the given integer value    
    """
    # create using dictionary
    _val = {"x":"rb_marker_m" + str(val) + "_x",
            "y":"rb_marker_m" + str(val) + "_y",
            "z":"rb_marker_m" + str(val) + "_z",}

    return _val