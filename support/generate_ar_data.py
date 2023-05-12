import pandas as pd
import numpy as np
import cv2
import msgpack as mp
import msgpack_numpy as mpn
import glob
import os
from cv2 import aruco


"""
these are default data, you can change it by calling the functions below

these functions were initially written for MIRA split videos protocol

you can use it for single file using *_sf functions, which is for a single file only
"""

_pth = os.path.dirname(os.getcwd())

calib_pth = os.path.join(_pth,"support", "AR_CALIBRATION.msgpack")
_calib_file = open(calib_pth, "rb")
unpacker = mp.Unpacker(_calib_file, object_hook=mpn.decode)
_calib = []
for unpacked in unpacker:
    _calib.append(unpacked)

cameraMatrix = _calib[0][0]
distCoeffs = _calib[0][1]
_calib_file.close()


def camera_parameters(ar_parameters = None, ar_dictionary = None, markerLength = 0.05, markerSeparation = 0.01):

    """
    ar_parameters: aruco camera parameters using 'aruco.DetectorParameters_create()'
    ar_dictionary: dictionary of aruco markers
    markerLength: length of marker in meters
    markerSeparation: separation between markers in meters
    """

    ARUCO_PARAMETERS = ar_parameters
    ARUCO_DICT = ar_dictionary

    if ar_parameters is None:
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    if ar_dictionary is None:
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    
    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=1,
            markersY=1,
            markerLength=markerLength,
            markerSeparation=markerSeparation,
            dictionary=ARUCO_DICT)

    return ARUCO_PARAMETERS, ARUCO_DICT, board


def estimate_ar_pose(frame, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,is_color= False, ar_params = None, ar_dict = None, board = None):

    """
    frame: frame to be processed
    cameraMatrix: camera matrix from calibration
    distCoeffs: distortion coefficients from calibration file
    """

    ARUCO_PARAMETERS = ar_params
    ARUCO_DICT = ar_dict

    if is_color:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # ARUCO_PARAMETERS, ARUCO_DICT, board = camera_parameters()
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Refine detected markers
    # Eliminates markers not part of our board, adds missing markers to the board
    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = gray,
            board = board,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            cameraMatrix = cameraMatrix,
            distCoeffs = distCoeffs)

    rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
    return rotation_vectors, translation_vectors, _objPoints, ids


def get_ar_pose_data(_pth, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, process_raw = False, is_color = True, single_file=False, flip_frame = False, _pth_to_save="", print_display = False):

    """
    _pth: path to video (msgpack) files containing calibration data
    cameraMatrix: camera matrix from calibration
    distCoeffs: distortion coefficients from calibration file
    process_raw: if True, it will process raw data, if False, it will process processed data
    is_color: if True, it will process color data, if False, it will process grayscale data
    single_file: if True, it will process a single file, if False, it will process all files in the folder
    flip_frame: if True, it will flip the frame, if False, it will not flip the frame
    _pth_to_save: path to save the processed data
    """
    df = pd.DataFrame(columns=["frame_id", "x", "y", "z", "yaw", "pitch", "roll"])
    rotation_vectors, translation_vectors = None, None

    ar_params, ar_dict, board = camera_parameters()
    _is_color = is_color

    if process_raw:
        targetPattern = f"{_pth}\\COLOUR*"
        color_file_list = glob.glob(targetPattern)


        for fname in color_file_list:
            cfile = open(fname, "rb") #colour file
            unpacker = mp.Unpacker(cfile, object_hook=mpn.decode)
            for frame in unpacker:

                rotation_vectors, translation_vectors, _, ids = estimate_ar_pose(frame, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, is_color = _is_color, ar_params = ar_params, ar_dict = ar_dict, board = board)
                if ids is not None:
                    data = [ids[0][0]]
                else:
                    data = [np.nan]

                if rotation_vectors is not None and rotation_vectors is not []:
                    data.extend(translation_vectors[0][0])
                    data.extend(rotation_vectors[0][0])
                    df.loc[len(df)] = data
                else:
                    data.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    df.loc[len(df)] = data
                
            cfile.close()
    elif not single_file:

        vid_pth = os.path.join(_pth, "Video.avi")
        cap = cv2.VideoCapture(vid_pth)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rotation_vectors, translation_vectors, _ , ids = estimate_ar_pose(frame, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

                data = []
                if rotation_vectors is not None:
                    data.extend(translation_vectors[0][0])
                    data.extend(rotation_vectors[0][0])
                    df.loc[len(df)] = data
                else:
                    data.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    df.loc[len(df)] = data
                
                rotation_vectors = None
                translation_vectors = None
                
            else:
                break
        cap.release()
    else:

        cfile = open(_pth, "rb") #colour file
        unpacker = mp.Unpacker(cfile, object_hook=mpn.decode)
        for frame in unpacker:
            if flip_frame:
                frame = cv2.flip(frame, 1)
            rotation_vectors, translation_vectors, _, ids = estimate_ar_pose(frame, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, ar_params = ar_params, ar_dict = ar_dict, board = board)
            data = []
            if ids is not None:
                data = [ids[0][0]]
            else:
                data = [np.nan]

            if rotation_vectors is not None and rotation_vectors is not []:
                data.extend(translation_vectors[0][0])
                data.extend(rotation_vectors[0][0])
                df.loc[len(df)] = data
            else:
                data.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                df.loc[len(df)] = data
            
        cfile.close()
    if print_display:
        print("returning dataframe")
    return df

def detect_ar_markers(frame, ar_params = None, ar_dict = None, board = None):

    """
    frame: frame to be processed
    cameraMatrix: camera matrix from calibration
    distCoeffs: distortion coefficients from calibration file
    """

    ARUCO_PARAMETERS = ar_params
    ARUCO_DICT = ar_dict

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ARUCO_PARAMETERS, ARUCO_DICT, board = camera_parameters()
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    return corners, ids, rejectedImgPoints

def add_time_col(df, _pth):
    
    targetPattern = f"{_pth}\\PARAM*"
    param_file = glob.glob(targetPattern)[0]

    # Read in the param file
    with open(param_file, "rb") as f:
        unpacker = mp.Unpacker(f, object_hook=mpn.decode)
        _tmp = []
        for counter, _obj in enumerate(unpacker):
            if counter >1:

                _tmp.append(_obj[0])
            
        df["time"] = _tmp
    
    return df





if __name__ == '__main__':
    # print("hi")
    try:
        get_ar_pose_data(r"C:\Users\CMC\Dropbox\mira\mira_vellore\splitVideos\SUJIXXXXXXXXU010120000000XXXXXXXXX\test_trial_0", process_raw=True)
    except:
        pass
    print("in a different computer")