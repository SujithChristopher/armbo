import cv2
import numpy as np
import os
import sys

from cv2 import aruco
from support.ar_calculations import calculate_rotmat

def detect_lframe_from_img(img):
    """
    Detects the lframe from the image

    Idx: 6 - zvector
    Idx: 9 - orgin
    Idx: 10 - xvector
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    return corners, ids, rejectedImgPoints

def calculate_rotmat_from_3markers(corners, ids, camera_matrix, dist_coeffs, marker_length = 0.05):

    rotation_vector, translation_vector, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    ids = list(ids)

    z_inx = ids.index(6)
    org_inx = ids.index(9)
    x_inx = ids.index(10)
    # print(z_inx, org_inx, x_inx, flush = True)

    zvec = translation_vector[z_inx][0]
    zvec = np.reshape(zvec, (3, 1))
    org = translation_vector[org_inx][0] 
    org = np.reshape(org, (3, 1))
    xvec = translation_vector[x_inx][0]
    xvec = np.reshape(xvec, (3, 1))

    rotMat = calculate_rotmat(xvec, zvec, org)
    t_xvec = zvec - org 
    t_zvec = xvec - org
    _z = rotMat.T@t_zvec
    _x = rotMat.T@t_xvec

    # print(_x.T[0], _z.T[0], end="\r")


    return [_x.T[0], _z.T[0]]