import numpy as np
from scipy import integrate

def set_zero(df, column_name = ["e_t", "e_rr", "e_rl"]):
    """
    Set the value of the column to 0
    """
    """resetting cart values to zero"""

    for i in column_name:
        df[i] = df[i] - df[i][0]

    return df
    

def get_angular_velocity(df, column_name = ["e_t", "e_rr", "e_rl"], ang_per_increment = 0.09, del_t = 0.01):
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


def get_directional_velocity(df, column_name, radius = 1, x = 1, y = 1):

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


    # mat = np.array([[-y, 1, 0], [-x, 0, -1], [x, 0, -1]])
    mat = np.array([[-y, 1, 0], [-x, 0, -1], [x, 0, -1]])
    pmat = np.linalg.pinv(mat)
    _vx = []
    _vy = []
    _w = []

    for i in range(len(df)):
        val = np.array([df[column_name[0]].iloc[i], df[column_name[1]].iloc[i], df[column_name[2]].iloc[i]]).reshape(3,1)
        res = np.dot(pmat, val) * radius
        _w.append(res[0][0])
        _vx.append(res[1][0])
        _vy.append(res[2][0])
    df["w"] = _w
    df["vx"] = _vx
    df["vy"] = _vy

    return df, df.columns

def get_position(df):
    """
    Calculate the position of the robot

    df should have "vx", "vy", "w" columns to calculate the position
    """

    _xval = df["vx"].cumsum()*0.01*0.5
    _yval = df["vy"].cumsum()*0.01*0.5

    df["x_val"] = _xval
    df["y_val"] = _yval
    
    return df, ["x_val", "y_val"]


def get_orientation(df, column_name = "w"):

    """
    Calculate the angle of the chasis, with respect to initial frame

    df should have "w" column to calculate the angle
    """

    if not column_name:
        column_name = "w"
    
    _angle = df[column_name].cumsum()*0.01

    df["theta"] = _angle

    return df, ["theta"]

def get_orientation_dt(df, column_name = "w", dt = []):

    """
    Calculate the angle of the chasis, with respect to initial frame

    df should have "w" column to calculate the angle
    """

    if not column_name:
        column_name = "w"
    
    _angle = integrate.cumtrapz(df[column_name], dt, initial=0)

    return df, ["theta"]