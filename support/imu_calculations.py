
from scipy.signal import savgol_filter
import polars as pl

def calculate_imu_orientation(df, filter_size=51, filter_order=1, zero_drift=50, dt = 0.01):
    """
    df: the dataframe containing the imu data
    filter_size: the size of the filter window (should be odd)
    The funciton uses the savgol filter to smooth the data
    filter_order: the order of the filter
    zero_drift: the number of rows to use to calculate the zero drift

    the df should contain gx, gy, gz columns

    returns: the dataframe with the calculated angles
    """    
    
    inst = True

    if not isinstance(df, pl.DataFrame):
        inst = False 
        _imu_df = df.copy()
        df = pl.from_pandas(df)

    else:
        _imu_df = df

    val_x = _imu_df["gx"][:zero_drift].mean()
    val_y = _imu_df["gy"][:zero_drift].mean()
    val_z = _imu_df["gz"][:zero_drift].mean()

    # subtracting the zero drift
    _imu_df = _imu_df.with_columns([
                                pl.col("gx") - val_x, 
                                pl.col("gy") - val_y,
                                pl.col("gz") - val_z
                            ])
    
    # smoothing the data using savgol filter
    if filter_order and filter_size is not None:
        _imu_df = _imu_df.with_columns([
                                pl.col("gx").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode(),
                                pl.col("gy").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode(),
                                pl.col("gz").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode()
                            ])
        _imu_df = _imu_df.drop_nulls()

    # calculating linear velocity from ax, ay, az
    _imu_df = _imu_df.with_columns([
                                (pl.col("gx").cumsum()*dt).alias("ang_x"),
                                (pl.col("gy").cumsum()*dt).alias("ang_y"),
                                (pl.col("gz").cumsum()*dt).alias("ang_z")
                            ])

    if not inst:
        return _imu_df.to_pandas()

    return _imu_df

def calculate_imu_velocity(df, filter_size=None, filter_order=None, zero_drift=50, dt = 0.01):
    """
    df: the dataframe containing the imu data
    filter_size: the size of the filter window (should be odd)
    The funciton uses the savgol filter to smooth the data
    filter_order: the order of the filter
    zero_drift: the number of rows to use to calculate the zero drift

    the df should contain ax, ay, az columns

    returns: the dataframe with the calculated angles
    """    
    
    inst = True

    if not isinstance(df, pl.DataFrame):
        inst = False 
        _imu_df = df.copy()
        df = pl.from_pandas(df)

    else:
        _imu_df = df

    val_x = _imu_df["ax"][:zero_drift].mean()
    val_y = _imu_df["ay"][:zero_drift].mean()
    val_z = _imu_df["az"][:zero_drift].mean()


    # subtracting the zero drift
    _imu_df = _imu_df.with_columns([
                                pl.col("ax") - val_x, 
                                pl.col("ay") - val_y,
                                pl.col("az") - val_z
                            ])
    
    # smoothing the data using savgol filter
    if filter_order and filter_size is not None:
        _imu_df = _imu_df.with_columns([
                                pl.col("ax").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode(),
                                pl.col("ay").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode(),
                                pl.col("az").map(lambda x: savgol_filter(x.to_numpy(), filter_size, filter_order)).explode()
                            ])
        _imu_df = _imu_df.drop_nulls()

    # calculating linear velocity from ax, ay, az
    _imu_df = _imu_df.with_columns([
                                (pl.col("ax").cumsum()*dt).alias("vx"),
                                (pl.col("ay").cumsum()*dt).alias("vy"),
                                (pl.col("az").cumsum()*dt).alias("vz")
                            ])

    if not inst:
        return _imu_df.to_pandas()

    return _imu_df


def calculate_imu_displacement(df, filter_size=None, filter_order=None, zero_drift=0, dt = 0.01):
    """
    df: the dataframe containing the imu data
    filter_size: the size of the filter window (should be odd)
    The funciton uses the savgol filter to smooth the data
    filter_order: the order of the filter
    zero_drift: the number of rows to use to calculate the zero drift

    the df should contain ax, ay, az columns

    returns: the dataframe with the calculated displacemnt
    """    
    
    inst = True

    if not isinstance(df, pl.DataFrame):
        inst = False 
        _imu_df = df.copy()
        df = pl.from_pandas(df)
    else:
        _imu_df = df

    if filter_order and filter_size is not None:
        _imu_df = _imu_df.with_columns([
                                pl.col("ax").map(lambda x: savgol_filter(x, filter_size, filter_order)),
                                pl.col("ay").map(lambda x: savgol_filter(x, filter_size, filter_order)),
                                pl.col("az").map(lambda x: savgol_filter(x, filter_size, filter_order))
                            ])

    # calculating angle from ax, ay, az
    my_dict = {"disp_x":[],"disp_y":[],"disp_z":[]}

    for i in range(len(_imu_df)):
        if i == 0:
            my_dict["disp_x"].append(0)
            my_dict["disp_y"].append(0)
            my_dict["disp_z"].append(0)
        else:

            my_dict["disp_x"].append(my_dict["disp_x"][i-1] + (_imu_df["vel_x"][i]) * dt)
            my_dict["disp_y"].append(my_dict["disp_y"][i-1] + (_imu_df["vel_y"][i]) * dt)
            my_dict["disp_z"].append(my_dict["disp_z"][i-1] + (_imu_df["vel_z"][i]) * dt)

    df = df.to_pandas()
    _disp_df = df
    _disp_df["disp_x"] = my_dict["disp_x"]
    _disp_df["disp_y"] = my_dict["disp_y"]
    _disp_df["disp_z"] = my_dict["disp_z"]

    return _disp_df
