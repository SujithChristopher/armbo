import itertools
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

"""
funciton list

    combinations_of_pseudo_mat()
    interpolate_data()
    calculate_difference()
    calculate_directional_velocities()

"""

def combinations_of_pseudo_mat(_b1, _b2, _b3, _b4, _a1, _a2, _a3, _a4, radius):

    """
    This function calculates all possible combinations of input arguments
    returns a ndarray of pseudo matrices of all possible combinations
    """
    
    iterables = list(itertools.product(_b1, _b2, _b3, _b4, _a1, _a2, _a3, _a4))
    length_of_iterables = len(iterables)
    pseudo_mat = np.empty((length_of_iterables, 3, 4))
    iter_array = np.array(iterables)

    for idx, _arg in enumerate(itertools.product(_b1, _b2, _b3, _b4, _a1, _a2, _a3, _a4)):
        
        a1 = np.radians(_arg[4])
        a2 = -np.radians(_arg[5])
        a3 = np.radians(_arg[6] + 90)
        a4 = -np.radians(_arg[7] + 90)

        b1 = np.radians(_arg[0])
        b2 = -np.radians(_arg[1])
        b3 = np.radians(_arg[2])
        b4 = -np.radians(_arg[3])

        g1 = -np.pi/4
        g2 = np.pi/4
        g3 = g2
        g4 = g1
        
        li = 101.36/1000

        t = (-1/radius)*np.array([[np.cos(b1 - g1)/ np.sin(g1), np.sin(b1 - g1)/np.sin(g1), li * np.sin(b1 - g1 - a1)/np.sin(g1)],
                            [np.cos(b2 - g2)/ np.sin(g2), np.sin(b2 - g2)/np.sin(g2), li * np.sin(b2 - g2 - a2)/np.sin(g2)],
                            [np.cos(b3 - g3)/ np.sin(g3), np.sin(b3 - g3)/np.sin(g3), li * np.sin(b3 - g3 - a3)/np.sin(g3)],
                            [np.cos(b4 - g4)/ np.sin(g4), np.sin(b4 - g4)/np.sin(g4), li * np.sin(b4 - g4 - a4)/np.sin(g4)]]
                            )
        
        pseudo_t = np.linalg.pinv(t)
        pseudo_mat[idx] = pseudo_t
        return pseudo_mat, iter_array, length_of_iterables

def interpolate_data(target, reference, columns):

    """
    this function interpolates the dataframe target to the dataframe reference
    input is the dataframe target and reference
    both should have "time" column to them
    """

    if target.index.values[0] != reference.index.values[0]:
        target = target.reset_index(drop=True)

    if target["time"].dtype != reference["time"].dtype:

        target["time"] = pd.to_datetime(target["time"])
        reference["time"] = pd.to_datetime(reference["time"])
        
        warnings.warn("time column type is not the same, converting to datetime")

    _temp_df = pd.DataFrame(columns=columns)

    for i in columns:
        inp_fn = interp1d(pd.to_numeric(target["time"]), target[i], axis=0 ,fill_value='extrapolate')
        _temp_df[i] = inp_fn(pd.to_numeric(reference["time"]))

    _temp_df = _temp_df.fillna(0)
    _temp_df["time"] = reference["time"]
    return _temp_df

def iterate_get_diff(pseudo_mat, wheels, wheel_columns, mc, mc_cols):

    """
    iterate through pseudo_mat calculate the wheel coordinates for each iteration of pseudo_mat 
    and get the difference between the wheels and the mc for each iteration
    """

    ln, _, _ = pseudo_mat.shape

    for i in range(ln):
        _temp_df = calculate_directional_velocities(pseudo_mat[i], wheels, col_names=wheel_columns)

        for j in range(2):
            diff = mc[mc_cols[0]] - _temp_df["x"]
            print(diff)
        break
    pass



def calculate_directional_velocities(pseudo_t, df, col_names):

    """
    this function calculates the directional velocities of the wheels
    input is the matrix of pseudo-inverse of the transformation matrix

    returns a ndarray of the directional velocities
    """
    
    _temp_df = pd.DataFrame(columns=["vx", "vy", "w"])
    _val = []

    for i in range(len(df)):

        """
        this loop iterates through all rows of the dataframe for each pseudo_mat
        and calculates the new velocities
        """
        _v = pseudo_t @ np.array(df[col_names].iloc[i])
        _val.append(_v.T)

    _temp_df[["vx", "vy", "w"]] = _val

    _temp_df = calculate_displacement(_temp_df)
    return _temp_df



def calculate_displacement(df):
    
    """
    calculating displacement
    s=(1/2)* (v+u)t
    v = current velocityn
    u = initial velocity
    t = time
    s = displacement
    """
    _xval = []
    _yval = []
    xf_disp = 0
    yf_disp = 0
    for i in range(len(df)):
        if i == 0:
            _xval.append(0)
            _yval.append(0)
        else:
            x_disp = 0.5*(df["vx"].iloc[i] + df["vx"].iloc[i-1])*0.01
            y_disp = 0.5*(df["vy"].iloc[i] + df["vy"].iloc[i-1])*0.01
            # print(y_disp)
            xf_disp = xf_disp+x_disp
            yf_disp = yf_disp+y_disp
            _xval.append(xf_disp)
            _yval.append(yf_disp)

    df["x"] = _xval
    df["y"] = _yval

    return df

def transform_coordinates(df, rotmat, org):

    for i in range(len(df)):
        val = df[["x", "y", "z"]].loc[i].values
        val = np.reshape(val, (3,1))
        _temp = val + org
        _val = rotmat.T @ _temp
        df[["x", "y", "z"]].loc[i] = _val.T[0]
    return df





