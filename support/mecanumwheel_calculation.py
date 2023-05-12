import numpy as np
import polars as pl

def set_zero(df, column_name = ["e_fr", "e_fl", "e_rr", "e_rl"]):
    """
    Set the value of the column to 0
    """
    """resetting cart values to zero"""
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False 

    for i in column_name:
        df = df.with_columns([pl.col(i).apply(lambda x: x - df[i][0])])

    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()
    return df
    

def get_angular_velocity(df, column_name = ["e_fr", "e_fl", "e_rr", "e_rl"], ang_per_increment = 0.15, del_t = 0.01):
    """
    Calculate the angular velocity of the robot
    """
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False 

    # Calculate the angular velocity
    for name in column_name:
        df = df.with_columns([pl.col(name).apply(lambda x: x * ang_per_increment).alias(name + "_angle")])
        df = df.with_columns([((pl.col(name + "_angle").diff()/ del_t)*np.pi/180).alias(name + "_av")])
        # replace null values with 0
        df = df.fill_null(0)
    
    _ang_column = []
    for i in column_name:
        _ang_column.append(i + "_av")

    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()
    return df, _ang_column


def get_directional_velocity(df, column_name, radius = 1, l = 1, w = 1):

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
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False 

    mat = np.array([[-l-w , 1, -1], [l+w, 1, 1], [l+w, 1, -1],[-l-w, 1, 1]]) # matrix for calculating the directional velocity
    pmat = np.linalg.pinv(mat)

    my_dict = {"_vx":[],"_vy":[],"_w":[]}

    for i in range(len(df)):
        val = df[column_name][i].to_numpy().reshape(4,1)
        res = np.dot(pmat, val) * radius
        my_dict["_w"].append(res[0][0])
        my_dict["_vx"].append(res[1][0])
        my_dict["_vy"].append(res[2][0])

    # add the calculated values to the dataframe
    df = df.with_columns([pl.Series(name = "vx", values = my_dict["_vx"]),
                        pl.Series(name = "vy", values = my_dict["_vy"]),
                            pl.Series(name = "w", values = my_dict["_w"])])
    
    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()

    return df, df.columns

def get_position(df, dt = 0.01):
    """
    Calculate the position of the robot

    df should have "vx", "vy", "w" columns to calculate the position
    """
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False 
    # # calculate the cumulative sum of the values multiplied by dt
    
    # df = df.with_columns([(pl.col("vx").cumsum() * dt *0.5).alias("x"),
    #                         (pl.col("vy").cumsum() * dt *0.5).alias("y")])
    
    df = df.with_columns([(pl.col("vx").cumsum() * dt).alias("x"),
                            (pl.col("vy").cumsum() * dt).alias("y")])
    
    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()
    
    return df, ["x", "y"]


def get_orientation(df, dt = 0.01, column_name = "w"):

    """
    Calculate the angle of the chasis, with respect to initial frame

    df should have "w" column to calculate the angle
    """
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False

    if not column_name:
        column_name = "w"

    # calculate the cumulative sum of the values multiplied by dt

    df = df.with_columns([(pl.col("w").cumsum() * dt).alias("theta")])


    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()

    return df, ["theta"]


def get_orientation_dt(df, column_name = "w", dt = []):

    """
    Calculate the angle of the chasis, with respect to initial frame

    df should have "w" column to calculate the angle
    """
    inst = True

    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
        inst = False

    if not column_name:
        column_name = "w"

    my_dict = {"_theta":[]}
    angle = 0
    for i in range(len(df[column_name])):

        if i == 0:
            my_dict["_theta"].append(0)
        else:
            angle = angle + (df[column_name][i] + df[column_name][i-1])*dt[i]
            my_dict["_theta"].append(angle)

    # add the calculated values to the dataframe
    df = df.with_columns([pl.Series(name = "theta", values = my_dict["_theta"])])

    if not inst: # if the input is not a polars dataframe, convert it to pandas dataframe when returning
        df = df.to_pandas()

    return df, ["theta"]