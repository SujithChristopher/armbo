import ctypes
import glob
import os
import pickle
from ctypes import c_ushort

import msgpack as mp
import msgpack_numpy as mpn

import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import _CameraSpacePoint

# from support_py import *
from mapper import color_2_world

_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)


def camspace(pth, res_s=False):
    cam_list = []
    targetPattern = f"{pth}\\DEPTH*"
    dep_lst = glob.glob(targetPattern)

    for i in dep_lst:
        a = os.path.basename(i)
        dir_pth = os.path.dirname(i)
        a = a.replace("DEPTH", "CAMSPACE")
        cam_list.append(f"{dir_pth}//{a}")

    targetPattern_param = f"{pth}\\PARAM*"
    param_file_name = glob.glob(targetPattern_param)

    param_file = open(param_file_name[0], "rb")

    unpacker = mp.Unpacker(param_file, object_hook=mpn.decode)
    prm = []
    for unpacked in unpacker:
        prm.append(unpacked)

    xypos = prm[0]
    # xypos = pickle.load(param_file)

    xPos = int(xypos[0] / 2)
    yPos = int(xypos[1] / 2)
    yRes = 736
    xRes = 864
    _a = prm[1]

    col_sz = len(dep_lst)

    kinectColor = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    kinectDepth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

    for i in range(col_sz):
        cam_sp = open(cam_list[i], "wb")
        dep_file = open(dep_lst[i], "rb")
        unp_dep = mp.Unpacker(dep_file, object_hook=mpn.decode)
        dep = []
        for unpacked in unp_dep:
            dep.append(unpacked)
        l1 = 0
        l2 = len(dep) - 1
        rn = col_sz
        print(f"Processing files {i+1} out of {rn}")

        while True:
            if kinectColor.has_new_color_frame() and kinectDepth.has_new_depth_frame():
                try:
                    # depthFrame = pickle.load(dep_file)
                    depthFrame = dep[l1]
                    l1 += 1
                    bb = ctypes.cast(depthFrame.ctypes, ctypes.POINTER(c_ushort))
                    depthData = color_2_world(_kinect, bb, _CameraSpacePoint, True)
                    depXYZraw = depthData[yPos * 2:yPos * 2 + yRes, xPos * 2:xPos * 2 + xRes].copy()
                    # depXYZraw = cv2.flip(depXYZraw, 1)
                    if res_s:
                        depXYZ = depXYZraw
                    else:
                        depXYZ = cv2.resize(depXYZraw, (432, 368))

                    depINT = np.int16(depXYZ * 1000)

                    cam_bin = mp.packb(depINT, default=mpn.encode)
                    cam_sp.write(cam_bin)
                    if l1 == l2 + 1:
                        dep_file.close()
                        cam_sp.close()
                        os.remove(dep_lst[i])
                        break

                except EOFError:
                    dep_file.close()
                    cam_sp.close()
                    os.remove(dep_lst[i])
                    break

    return "camera space processed"

    param_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camspace(r"C:\Users\CMC\Dropbox\mira\mira_vellore\splitVideos\SUJIXXXXXXXXU010120000000XXXXXXXXX\calibration")
