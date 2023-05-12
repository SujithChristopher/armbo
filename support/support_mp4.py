"""
Module with all the supporting classes and function for reading
and analysing Kinect data using OpenPose.

Author: Sivakumar Balasubramanian
Date: 04 Feb 2021
"""

import pickle
import sys
import numpy as np
import math
import cv2
from fpdf import FPDF


def read_pickle_file(fname):
    data = []
    with (open(fname, "rb")) as openfile:
        while True:
            try:
                data.append(pickle.load(openfile))
            except EOFError:
                break
    return data


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def read_colour_frames(fname):
    data = []
    cap = cv2.VideoCapture(fname)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            data.append(frame)
        else:
            break
    cap.release()
    return data


class Params:
    root = 'C:\\Users\\CMC\\Documents\\LVS\\data'
    outdir = 'outdir'
    diag = 'diagdata'
    body = 'bodydata'
    otdata = '_mocap'

    PoseJoints = ("HEAD", "NECK", "RSHD", "RELB", "RWRT",
                  "LSHD", "LELB", "LWRT", "RHIP", "RKNE",
                  "RANK", "LHIP", "LKNE", "LANK",
                  "REYE", "REAR", "LEYE", "LEAR")
    ExtraPoseJoints = ("TRNK",)
    MaxHumans = 1
    PerfHeader = ("subj", "date", "time", "session", "rwrst5_x", "rwrst5_y",
                  "rwrst5_z", "rwrst50_x", "rwrst50_y",
                  "rwrst50_z", "rwrst95_x", "rwrst95_y",
                  "rwrst95_z", "lwrst5_x", "lwrst5_y",
                  "lwrst5_z", "lwrst50_x", "lwrst50_y",
                  "lwrst50_z", "lwrst95_x", "lwrst95_y",
                  "lwrst95_z", "trnkx5", "trnky5",
                  "trnkz5", "rsfe5", "lsfe5", "rsaa5",
                  "lsaa5", "refe5", "lefe5")

    @staticmethod
    def get_pose_header_for_col():
        # Genrate header for pose in color space.
        _head = [f"{_pj}_{_v}_{_nh}"
                 for _nh in range(Params.MaxHumans)
                 for _pj in Params.PoseJoints
                 for _v in ("x", "y", "s")]
        _head = _head + [f"{_pj}_{_v}_{_nh}"
                         for _nh in range(Params.MaxHumans)
                         for _pj in Params.ExtraPoseJoints
                         for _v in ("x", "y", "s")]
        return _head

    @staticmethod
    def get_pose_header_for_dep():
        # Genrate header for pose in depth space.
        _head = [f"{_pj}_{_v}_{_nh}"
                 for _nh in range(Params.MaxHumans)
                 for _pj in Params.PoseJoints
                 for _v in ("x", "y", "s")]
        _head = _head + [f"{_pj}_{_v}_{_nh}"
                         for _nh in range(Params.MaxHumans)
                         for _pj in Params.ExtraPoseJoints
                         for _v in ("x", "y", "s")]
        return _head

    @staticmethod
    def get_pose_header_for_cam():
        # Genrate header for pose camera space.
        return [f"{_pj}_{_v}_{_nh}"
                for _nh in range(Params.MaxHumans)
                for _pj in Params.PoseJoints
                for _v in ("x", "y", "z", "s")]

    @staticmethod
    def get_joint_angles_header():
        joints = ("TRUNK_X", "TRUNK_Y", "TRUNK_Z", "RSHD_FE", "RSHD_AA",
                  "RELB_FE", "LSHD_FE", "LSHD_AA", "LELB_FE")
        return [f"{_pj}_{_nh}"
                for _nh in range(Params.MaxHumans)
                for _pj in joints]


def estimate_pose(coldata, pose_est):
    # Cycle through all the colorframe images and estimate pose.
    inx = 0
    N = len(coldata)
    humanpose = []
    while inx < N:
        # Read file.
        sys.stdout.write(f"\r{inx}")
        image = coldata[inx]
        h, w, _ = np.shape(image)

        # Estimate pose
        humans = pose_est.estimate_pose(image)
        humanpose.append(humans)
        inx += 1

    return humanpose


def estimate_pose_frame(coldata, pose_est):
    # single frame pose estimation
    humans = pose_est.estimate_pose(coldata)
    return humans


def get_poserow(hum, PoseJoints, ExtraPoseJoints):
    # Get pose for PoseJoints
    _rw = {}
    _nh = 0
    for _nj, _pj in enumerate(PoseJoints):
        # Make sure the human data is available
        if hum[_nj] is not None:
            for _k, _v in hum[_nj].items():
                _rw[f"{_pj}_{_k}_{_nh}"] = _v
    # Add the trunk
    # Make sure the human data is available
    if hum[18] is not None:
        for _k, _v in hum[18].items():
            _rw[f"TRNK_{_k}_{_nh}"] = _v
    return _rw


def get_poserow_frame(hum, PoseJoints, ExtraPoseJoints):
    # Get pose for PoseJoints
    _rw = {}
    _nh = 0
    for _nj, _pj in enumerate(PoseJoints):
        # Make sure the human data is available
        if hum[0][_nj] is not None:
            for _k, _v in hum[0][_nj].items():
                _rw[f"{_pj}_{_k}_{_nh}"] = _v
    # Add the trunk
    # Make sure the human data is available
    if hum[0][18] is not None:
        for _k, _v in hum[0][18].items():
            _rw[f"TRNK_{_k}_{_nh}"] = _v
    return _rw


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_trunk_angle(camfdf):
    # Define trunk frame at each time instant.
    jnames = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "TRUNK"]
    N = len(camfdf)
    _R = [None] * N
    _R0 = None
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # x axis
        _v1 = joints["LEFT_SHOULDER"] - joints["RIGHT_SHOULDER"]
        print(joints["LEFT_SHOULDER"])
        print(joints["RIGHT_SHOULDER"])
        print(_v1)
        _x = _v1 / np.linalg.norm(_v1, 2)
        _v2 = joints["TRUNK"] - joints["RIGHT_SHOULDER"]
        _y = (_v2 - (_x.T @ _v2) * _x)
        _y = _y / np.linalg.norm(_y, 2)
        _z = np.array([np.cross(_x.T[0], _y.T[0])]).T
        _R[inx] = np.hstack([_x, _y, _z])
        # Assign initial trunk orientation.
        if _R0 is None and isRotationMatrix(_R[inx]):
            _R0 = _R[inx]
        # Compute turnk angle
        if _R0 is not None:
            _dR = _R0.T @ _R[inx]
            angles[inx] = (rotationMatrixToEulerAngles(_dR) * 180 / np.pi
                           if ~np.isnan(np.linalg.det(_dR)) else
                           [np.nan, np.nan, np.nan])
        else:
            angles[inx] = [np.nan, np.nan, np.nan]

    return np.array(angles)


def get_rightshoulder_rotmat(camfdf):
    # Define trunk frame at each time instant.
    jnames = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "TRUNK"]
    N = len(camfdf)
    _R = [None] * N
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # x axis
        _v1 = joints["LEFT_SHOULDER"] - joints["RIGHT_SHOULDER"]
        _x = _v1 / np.linalg.norm(_v1, 2)
        _v2 = joints["TRUNK"] - joints["RIGHT_SHOULDER"]
        _y = (_v2 - (_x.T @ _v2) * _x)
        _y = _y / np.linalg.norm(_y, 2)
        _z = np.array([np.cross(_x.T[0], _y.T[0])]).T
        _R[inx] = np.hstack([_x, _y, _z])
    return np.array(_R)


def get_leftshoulder_rotmat(camfdf):
    # Define trunk frame at each time instant.
    jnames = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "TRUNK"]
    N = len(camfdf)
    _R = [None] * N
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # x axis
        _v1 = joints["RIGHT_SHOULDER"] - joints["LEFT_SHOULDER"]
        _x = _v1 / np.linalg.norm(_v1, 2)
        _v2 = joints["TRUNK"] - joints["LEFT_SHOULDER"]
        _y = (_v2 - (_x.T @ _v2) * _x)
        _y = _y / np.linalg.norm(_y, 2)
        _z = np.array([np.cross(_x.T[0], _y.T[0])]).T
        _R[inx] = np.hstack([_x, _y, _z])
    return np.array(_R)


def get_rightshoulder_angle(camfdf):
    jnames = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
    N = len(camfdf)
    rsh_rmat = get_rightshoulder_rotmat(camfdf)
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # upper-arm vector
        _rua = joints["RIGHT_ELBOW"] - joints["RIGHT_SHOULDER"]
        _ruaproj = rsh_rmat[inx].T @ _rua

        # Flexion-Extension angle
        _fe = np.arctan2(-_ruaproj[2, 0], _ruaproj[1, 0]) * 180 / np.pi
        # Abduction-Adduction angle
        _aa = np.arctan2(-_ruaproj[0, 0], _ruaproj[1, 0]) * 180 / np.pi
        angles[inx] = [_fe, _aa]
    return np.array(angles)


def get_leftshoulder_angle(camfdf):
    jnames = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    N = len(camfdf)
    rsh_rmat = get_leftshoulder_rotmat(camfdf)
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # upper-arm vector
        _rua = joints["LEFT_ELBOW"] - joints["LEFT_SHOULDER"]
        _ruaproj = rsh_rmat[inx].T @ _rua
        # Flexion-Extension angle
        _fe = np.arctan2(_ruaproj[2, 0], _ruaproj[1, 0]) * 180 / np.pi
        # Abduction-Adduction angle
        _aa = np.arctan2(-_ruaproj[0, 0], _ruaproj[1, 0]) * 180 / np.pi
        angles[inx] = [_fe, _aa]
    return np.array(angles)


def get_rightelbow_angles(camfdf):
    jnames = ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
    N = len(camfdf)
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # x axis
        _v1 = joints["RIGHT_ELBOW"] - joints["RIGHT_SHOULDER"]
        _v1 = _v1 / np.linalg.norm(_v1, ord=2)
        _v2 = joints["RIGHT_WRIST"] - joints["RIGHT_ELBOW"]
        _v2 = _v2 / np.linalg.norm(_v2, ord=2)
        angles[inx] = np.arccos((_v1.T @ _v2)[0, 0]) * 180 / np.pi
    return angles


def get_leftelbow_angles(camfdf):
    jnames = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    N = len(camfdf)
    angles = [None] * N
    for inx in range(N):
        # Joint vectors
        joints = get_joint_vectors(camfdf, inx, jnames)
        # x axis
        _v1 = joints["LEFT_ELBOW"] - joints["LEFT_SHOULDER"]
        _v1 = _v1 / np.linalg.norm(_v1, ord=2)
        _v2 = joints["LEFT_WRIST"] - joints["LEFT_SHOULDER"]
        _v2 = _v2 / np.linalg.norm(_v2, ord=2)
        angles[inx] = np.arccos((_v1.T @ _v2)[0, 0]) * 180 / np.pi
    return angles


def get_vector(df, inx, joint):
    return np.array([[df.loc[inx, f"{joint}_X"],
                      df.loc[inx, f"{joint}_Y"],
                      df.loc[inx, f"{joint}_Z"]]]).T


def get_joint_vectors(df, inx, joints):
    return {j: get_vector(df, inx, j) for j in joints}


def get_average_position(camfdf, joints=("RIGHT_SHOULDER", "LEFT_SHOULDER", "NECK", "TRUNK")):
    refpos = []
    N = len(camfdf)
    for inx in range(N):
        _jpos = get_joint_vectors(camfdf, inx, joints)
        refpos.append(np.mean([v for k, v in _jpos.items()
                               if np.isnan(np.sum(v)) == False], axis=0))
    return np.array(refpos)


def get_wrist_pos_wrt_ref(camfdf):
    # Reference position.
    refpos = get_average_position(camfdf, joints=("RIGHT_SHOULDER", "LEFT_SHOULDER", "NECK"))
    rwrt_pos = []
    lwrt_pos = []
    N = len(camfdf)
    for inx in range(N):
        # Wrist position
        rwrt_pos.append(get_vector(camfdf, inx, "RIGHT_WRIST") - refpos[inx])
        lwrt_pos.append(get_vector(camfdf, inx, "LEFT_WRIST") - refpos[inx])
    return np.array(rwrt_pos), np.array(lwrt_pos)


def get_performance_row(camfdf, trnk, rshd, lshd, relb, lelb):
    # Compute data summaries
    rwrt_pos, lwrt_pos = get_wrist_pos_wrt_ref(camfdf)
    # Endpoint space ROM
    r5, r50, r95 = [np.nanpercentile(rwrt_pos, q=q, axis=0)[:, 0]
                    for q in (5, 50, 95)]
    l5, l50, l95 = [np.nanpercentile(lwrt_pos, q=q, axis=0)[:, 0]
                    for q in (5, 50, 95)]
    # Trunk angles
    tx5, tx50, tx95 = [np.nanpercentile(trnk, q=q, axis=0)[0]
                       for q in (5, 50, 95)]
    ty5, ty50, ty95 = [np.nanpercentile(trnk, q=q, axis=0)[1]
                       for q in (5, 50, 95)]
    tz5, tz50, tz95 = [np.nanpercentile(trnk, q=q, axis=0)[2]
                       for q in (5, 50, 95)]

    # Shoulder angles
    rsfe5, rsfe50, rsfe95 = [np.nanpercentile(rshd, q=q, axis=0)[0]
                             for q in (5, 50, 95)]
    lsfe5, lsfe50, lsfe95 = [np.nanpercentile(lshd, q=q, axis=0)[0]
                             for q in (5, 50, 95)]
    # Shoulder angles
    rsaa5, rsaa50, rsaa95 = [np.nanpercentile(rshd, q=q, axis=0)[1]
                             for q in (5, 50, 95)]
    lsaa5, lsaa50, lsaa95 = [np.nanpercentile(lshd, q=q, axis=0)[1]
                             for q in (5, 50, 95)]

    # Elbow angles
    refe5, refe50, refe95 = [np.nanpercentile(relb, q=q, axis=0)
                             for q in (5, 50, 95)]
    lefe5, lefe50, lefe95 = [np.nanpercentile(lelb, q=q, axis=0)
                             for q in (5, 50, 95)]

    return {"rwrst5_x": r5[0],
            "rwrst5_y": r5[1],
            "rwrst5_z": r5[2],
            "rwrst50_x": r50[0],
            "rwrst50_y": r50[1],
            "rwrst50_z": r50[2],
            "rwrst95_x": r95[0],
            "rwrst95_y": r95[1],
            "rwrst95_z": r95[2],
            "lwrst5_x": l5[0],
            "lwrst5_y": l5[1],
            "lwrst5_z": l5[2],
            "lwrst50_x": l50[0],
            "lwrst50_y": l50[1],
            "lwrst50_z": l50[2],
            "lwrst95_x": l95[0],
            "lwrst95_y": l95[1],
            "lwrst95_z": l95[2],
            "trnkx5": tx5, "trnkx50": tx50, "trnkx95": tx95,
            "trnky5": ty5, "trnky50": ty50, "trnky95": ty95,
            "trnkz5": tz5, "trnkz50": tz50, "trnkz95": tz95,
            "rsfe5": rsfe5, "rsfe50": rsfe50, "rsfe95": rsfe95,
            "lsfe5": lsfe5, "lsfe50": lsfe50, "lsfe95": lsfe95,
            "rsaa5": rsaa5, "rsaa50": rsaa50, "rsaa95": rsaa95,
            "lsaa5": lsaa5, "lsaa50": lsaa50, "lsaa95": lsaa95,
            "refe5": refe5, "refe50": refe50, "refe95": refe95,
            "lefe5": lefe5, "lefe50": lefe50, "lefe95": lefe95}


"""
new functions added on 31-03-2021
"""


def get_clickedtask(self):
    if self.uns.isChecked():
        what_clicked = "UNS"  # Unspecified
    elif self.res.isChecked():
        what_clicked = "RES"  # rest
    elif self.txt.isChecked():
        what_clicked = "TXT"  # Texting
    elif self.fld.isChecked():
        what_clicked = "FLD"  # Folding
    elif self.but.isChecked():
        what_clicked = "BUT"  # buttoning
    elif self.bot.isChecked():
        what_clicked = "SRW"  # opening bottle
    elif self.brs.isChecked():
        what_clicked = "RNG"  # brushing
    elif self.wrt.isChecked():
        what_clicked = "WRT"  # Writing
    elif self.drk.isChecked():
        what_clicked = "DRK"  # drinking
    elif self.phc.isChecked():
        what_clicked = "PHC"  # phone call
    elif self.wpt.isChecked():
        what_clicked = "WPT"  # wiping
    elif self.trn.isChecked():
        what_clicked = "TRN"  # Turn on switch
    elif self.wlk.isChecked():
        what_clicked = "WLK"  # walking
    elif self.zuz.isChecked():
        what_clicked = "ZUZ"  # Zipping and unzip
    elif self.eas.isChecked():
        what_clicked = "EAS"  # eating with spoon
    elif self.dwc.isChecked():
        what_clicked = "DWC"  # drink with tea cup
    elif self.tyh.isChecked():
        what_clicked = "TYH"  # tying hair
    elif self.tkm.isChecked():
        what_clicked = "TKM"  # Taking medicine
    elif self.snh.isChecked():
        what_clicked = "SNH"  # sanitize hands
    elif self.waf.isChecked():
        what_clicked = "WAF"  # washing face
    elif self.tyk.isChecked():
        what_clicked = "TYK"  # tying knot
    else:
        what_clicked = "UNS"

    return what_clicked


def generate_pdf(self):
    blocks_no = 20
    maximum_ang = 30

    fpdf = FPDF(orientation='P', unit='mm', format='A4')
    fpdf.add_page()
    fpdf.set_font("Arial", style='', size=14)
    fpdf.image('.//src//logo.jpg', x=40, y=10, w=20, h=20)
    fpdf.multi_cell(w=180, h=10, txt="Christian Medical College, Vellore", align="C")
    fpdf.set_font("Arial", style='', size=16)
    fpdf.multi_cell(w=180, h=10, txt="Department of Bioengineering", align="C")
    fpdf.set_font("Arial", style='', size=18)
    fpdf.multi_cell(w=200, h=10, txt="", align="C")
    fpdf.multi_cell(w=100, h=10, txt="Patient Name: XXXX YYYY", align="L")
    fpdf.multi_cell(w=100, h=10, txt="Patient Age\t:  36", align="L")
    fpdf.multi_cell(w=100, h=10, txt="Condition\t\t:  Stroke", align="L")
    fpdf.multi_cell(w=100, h=10, txt="Hospital ID   :  10000G", align="L")
    fpdf.multi_cell(w=100, h=10, txt="Locality       :  Vellore", align="L")
    fpdf.image('.//src//color.jpeg', x=120, y=40, w=60, h=60)

    fpdf.multi_cell(w=100, h=10, txt=f"", align="L")
    fpdf.multi_cell(w=100, h=10, txt=f"", align="L")
    fpdf.multi_cell(w=100, h=10, txt=f"", align="L")

    fpdf.set_font("times", style='', size=18)
    fpdf.multi_cell(w=100, h=10, txt=f"", align="L")
    fpdf.multi_cell(w=100, h=10, txt=f"", align="L")
    fpdf.multi_cell(w=200, h=10, txt=f"ANALYSIS REPORT", align="C")
    # fpdf.multi_cell(w=100, h=10, txt=f"Max min Trunk angle X:  {self.max_min[0][0]}, {self.max_min[0][1]} ", align="L")
    # fpdf.multi_cell(w=100, h=10, txt=f"Max min Trunk angle Y:  {self.max_min[1][0]}, {self.max_min[1][1]} ", align="L")
    # fpdf.multi_cell(w=100, h=10, txt=f"Max min Trunk angle Z:  {self.max_min[2][0]}, {self.max_min[2][1]} ", align="L")

    fpdf.image(".//src//figure.png", x=0, y=150, w=110, h=100)
    fpdf.image(".//src//figure_scatter.png", x=110, y=150, w=100, h=100)
    fpdf.output('.//src//report.pdf', 'F')
    return "pdf generated"

