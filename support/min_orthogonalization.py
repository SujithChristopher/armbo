"""
This program is a support for l_frame_save_multicam.py

this orthogonalizes the new l frame, and displayes the v1, v2 values
"""
import numpy as np
import cv2

from ..support.ar_calculations import calculate_rotmat

def orthogonalize(corners, ids, camera_matrix, dist_coeffs, marker_length = 0.05, subtext = ""):

    rotation_vector, translation_vector, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    z_inx = ids.index(6)
    org_inx = ids.index(9)
    x_inx = ids.index(10)

    zvec = translation_vector[z_inx][0]
    zvec = np.reshape(zvec, (3, 1))
    org = translation_vector[org_inx][0] 
    org = np.reshape(org, (3, 1))
    xvec = translation_vector[x_inx][0]
    xvec = np.reshape(xvec, (3, 1))

    rotMat = calculate_rotmat(xvec, zvec, org)

    translation_correction = np.array([0.045, -0.05, 0.045]).reshape(3, 1) # adding the corrections in the new L frame
    
    t_zvec = zvec - org 
    t_xvec = xvec - org

    rotMat.T@t_xvec + translation_correction
    
    print(rotMat.T@t_xvec, "this is t_xvec", subtext)
    print(rotMat.T@t_zvec, "this is t_zvec", subtext)
    