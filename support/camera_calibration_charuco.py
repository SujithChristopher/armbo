import numpy as np
import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path
from tqdm import tqdm
import msgpack as mp
import msgpack_numpy as mpn
import glob
import os
import path
import sys

# directory reach
directory = os.path.realpath(__file__)
append_path = os.path.dirname(os.path.dirname(directory)) # go two folders up
  
# setting path
sys.path.append(append_path)

""" this program detectes the video frames and generates calibration file"""

def calibrate_camera_from_file(pth, pth_to_save="",saving_name = None, save_to_dir = False, cvt_color = True, markerLength=5, markerSeparation=1, drop_frames=50, img_flip = True):
    """
    this function calibrates from single msgpack file to
    markerLength = 5  # Here, measurement unit is centimetre.

    markerSeparation: separation between markers
    markerSeparation = 1   # Here, measurement unit is centimetre.

    drop_frames: number of detected frames to drop in percentage

    pth_to_save: path to save the calibration file

    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

    # create arUco board
    board = aruco.GridBoard_create(3, 4, markerLength, markerSeparation, aruco_dict)

    '''uncomment following block to draw and show the board'''
    #img = board.draw((864,1080))
    #cv2.imshow("aruco", img)

    arucoParams = aruco.DetectorParameters_create()

    color_file = open(pth, 'rb')
    color_frame = []
    unpacker = mp.Unpacker(color_file, object_hook=mpn.decode)
    for unpacked in unpacker:
        if img_flip:
            unpacked = cv2.flip(unpacked, 1)

        color_frame.append(unpacked)

    counter, corners_list, id_list = [], [], []
    first = True

    """ignoring some frames"""

    ln = len(color_frame)
    rnd = np.random.choice(ln, 150, replace=False)
    

    for idx, frame in enumerate(tqdm(color_frame)):

        if idx in rnd:
            if cvt_color:
                img_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            else:
                img_gray = frame

            corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
            try:
                if first == True:
                    corners_list = corners
                    id_list = ids
                    first = False
                else:
                    corners_list = np.vstack((corners_list, corners))
                    id_list = np.vstack((id_list,ids))
                counter.append(len(ids))
            except:
                continue

    print('Found {} unique markers'.format(np.unique(ids)))
    counter = np.array(counter)

    print("corners_list: ", corners_list.shape)

    print ("Calibrating camera .... Please wait...")

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print(" Camera matrix is \n", mtx, "\n And is stored in AR_CALIBRATION.msgpack file along with distortion coefficients : \n", dist)

    # save the calibration data
    if pth_to_save != "":

        if saving_name is None:
            saving_name = "AR_CALIBRATION.msgpack"
        else:
            saving_name = saving_name + ".msgpack"

        with open(os.path.join(os.path.dirname(pth_to_save), saving_name), 'wb') as f:
            pckd = mp.packb((mtx, dist), default=mpn.encode)
            f.write(pckd)
            f.close()
    elif save_to_dir:
        if saving_name is None:
            saving_name = "AR_CALIBRATION.msgpack"
        else:
            saving_name = saving_name + ".msgpack"

        with open(os.path.join(os.path.dirname(pth), saving_name), "wb") as p:
            pckd = mp.packb((mtx, dist), default=mpn.encode)
            p.write(pckd)
            p.close()
    else:
        return mtx, dist




def calibrate_camera(pth, pth_to_save="", markerLength=5, markerSeparation=1, drop_frames=50, img_flip = True):

    """
    pth: path to video (msgpack) files containing calibration data
    markerLength: length of the marker's side
    markerLength = 5  # Here, measurement unit is centimetre.

    markerSeparation: separation between markers
    markerSeparation = 1   # Here, measurement unit is centimetre.

    drop_frames: number of detected frames to drop in percentage

    pth_to_save: path to save the calibration file

    This decodes MIRA split videos protocol
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

    # create arUco board
    board = aruco.GridBoard_create(3, 4, markerLength, markerSeparation, aruco_dict)

    '''uncomment following block to draw and show the board'''
    #img = board.draw((864,1080))
    #cv2.imshow("aruco", img)

    arucoParams = aruco.DetectorParameters_create()

    targetPattern = f"{pth}\\COLOUR*"
    color_file_list = glob.glob(targetPattern)

    # read msgpack file
    color_frame = []
    first = True

    for idx, fname in enumerate(color_file_list):
        color_file = open(fname, 'rb')
        unpacker = mp.Unpacker(color_file, object_hook=mpn.decode)
        for unpacked in unpacker:
            if img_flip:
                unpacked = cv2.flip(unpacked, 1)

            color_frame.append(unpacked)

    counter, corners_list, id_list = [], [], []
    first = True

    """ignoring some frames"""

    ln = len(color_frame)
    rnd = np.random.choice(ln, 150, replace=False)
    

    for idx, frame in enumerate(tqdm(color_frame)):

        if idx in rnd:
            
            img_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
            try:
                if first == True:
                    corners_list = corners
                    id_list = ids
                    first = False
                else:
                    corners_list = np.vstack((corners_list, corners))
                    id_list = np.vstack((id_list,ids))
                counter.append(len(ids))
            except:
                continue

    print('Found {} unique markers'.format(np.unique(ids)))
    counter = np.array(counter)

    print("corners_list: ", corners_list.shape)

    print ("Calibrating camera .... Please wait...")

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print(" Camera matrix is \n", mtx, "\n And is stored in AR_CALIBRATION.msgpack file along with distortion coefficients : \n", dist)

    # save the calibration data
    if pth_to_save != "":
        with open(os.path.join(pth_to_save, "AR_CALIBRATION.msgpack"), 'wb') as f:
            pckd = mp.packb((mtx, dist), default=mpn.encode)
            f.write(pckd)
            f.close()
    else:
        with open(os.path.join(pth, "AR_CALIBRATION.msgpack"), "wb") as p:
            pckd = mp.packb((mtx, dist), default=mpn.encode)
            p.write(pckd)
            p.close()



if __name__ == "__main__":

    calibrate_camera(r"C:\Users\CMC\Dropbox\mira\mira_vellore\splitVideos\SUJIXXXXXXXXU010120000000XXXXXXXXX\calibration")