import os
import numpy as np
import pandas as pd
import sys
import os

def set_zero(df, column_name = ["e_fr", "e_fl", "e_rr", "e_rl"]):
    """
    Set the value of the column to 0
    """
    """resetting cart values to zero"""

    df[column_name[0]] = df[column_name[0]]- df[column_name[0]].iloc[0]
    df[column_name[1]] = df[column_name[1]]- df[column_name[1]].iloc[0]
    df[column_name[2]] = df[column_name[2]]- df[column_name[2]].iloc[0]
    df[column_name[3]] = df[column_name[3]]- df[column_name[3]].iloc[0]

    return df
    

def get_angular_velocity(df, column_name = ["e_fr", "e_fl", "e_rr", "e_rl"], ang_per_increment = 0.09, del_t = 0.01):
    """
    Calculate the angular velocity of the robot
    """

    # Calculate the angular velocity
    for name in column_name:
        df[name + "_av"] = (df[name] * ang_per_increment).diff()/ del_t
        df[name + "_av"] = df[name + "_av"]* np.pi/180
    df = df.fillna(0)

    _ang_column = []
    for i in column_name:
        _ang_column.append(i + "_av")
    
    return df, _ang_column


def get_directional_velocity(df, column_name, radius, lx, ly):

    """
    Calculate the directional velocity of the robot
    """

    """
    av_fr = column_name[0] front right
    av_fl = column_name[1] front left
    av_rr = column_name[2] rear right
    av_rl = column_name[3] rear left
    column names might vary, but the position of the wheel is fixed and should not change

    radius = radius of the wheel in meters

    radius = 47.5/1000
    lx = 79 #half of the distance between the wheels
    ly = 122.5/2
    """

    # Calculate the directional velocity
    df["vx"] = ( df[column_name[1]] + df[column_name[0]] + df[column_name[3]] + df[column_name[2]])*(radius/4)
    df["vy"] = (-df[column_name[1]] + df[column_name[0]] + df[column_name[3]] - df[column_name[2]])*(radius/4)
    df["w"] =  (-df[column_name[1]] + df[column_name[0]] - df[column_name[3]] + df[column_name[2]])*(radius/(4*(lx + ly)))

    return df, df[["vx", "vy", "w"]].columns.values

def get_position(df):
    """
    Calculate the position of the robot

    df should have "vx", "vy", "w" columns to calculate the position
    """

    _xval = []
    _yval = []
    xf_disp = 0
    yf_disp = 0
    for i in range(len(df["vx"])):
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

    df["x_val"] = _xval
    df["y_val"] = _yval
    
    return df, ["x_val", "y_val"]

def get_orientation(df):
    
    """
    Calculate the angle of the chasis, with respect to initial frame

    df should have "w" column to calculate the angle
    """
    
    _angle = []
    angle = 0
    for i in range(len(df["w"])):
        if i == 0:
            _angle.append(0)
        else:
            angle = angle + (df["w"].iloc[i] + df["w"].iloc[i-1])*0.01
            _angle.append(angle)

    df["theta"] = _angle

    return df, ["theta"]
