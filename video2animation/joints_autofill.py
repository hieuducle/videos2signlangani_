import os
import os.path as osp
import numpy as np

from smplifyx.data_parser import read_keypoints


mid_shoulder_id = 1
right_shoulder_id = 2
left_shoulder_id = 5
mid_hip_id = 8
right_hip_id = 9
left_hip_id = 12

base_mid_shoulder = np.array([639.127, 341.447, 0], dtype=np.float32)
base_right_shoulder = np.array([521.651, 345.319, 0], dtype=np.float32)
base_left_shoulder = np.array([758.716, 341.394, 0], dtype=np.float32)
base_mid_hip = np.array([641.173, 697.91, 0], dtype=np.float32)
base_right_hip = np.array([568.682, 697.933, 0], dtype=np.float32)
base_left_hip = np.array([711.662, 697.968, 0], dtype=np.float32)

base_shoulder_length = np.linalg.norm(base_mid_shoulder - base_right_shoulder) + np.linalg.norm(base_mid_shoulder - base_left_shoulder)
scale_mid_hip_from_mid_shoulder = (base_mid_hip - base_mid_shoulder) / base_shoulder_length
scale_right_hip_from_mid_hip = (base_right_hip - base_mid_hip) / base_shoulder_length
scale_left_hip_from_mid_hip = (base_left_hip - base_mid_hip) / base_shoulder_length


def fill_joints(pose_folder:str, use_hands:bool, use_face:bool):
    """
    Auto fill missing hip joint base on shoulder
    Joint index in keypoints object
        1: mid shoulder
        2: right shoulder
        5: left shoulder
        8: mid hip
        9: right hip
        12: left hip
    """

    pose_data = []
    for idx, keypoints_fn in enumerate(os.listdir(pose_folder)):
        keypoints = read_keypoints(osp.join(pose_folder, keypoints_fn), use_hands=use_hands, use_face=use_face)
        keypoints = np.stack(keypoints.keypoints)[[0]]

        right_shoulder_length = 0
        left_shoulder_length = 0
        if not np.array_equal(keypoints[0][right_shoulder_id][:2], [0,0]):
            right_shoulder_length = np.linalg.norm(keypoints[0][mid_shoulder_id][:2]-keypoints[0][right_shoulder_id][:2])
        if not np.array_equal(keypoints[0][left_shoulder_id][:2], [0,0]):
            left_shoulder_length = np.linalg.norm(keypoints[0][mid_shoulder_id][:2]-keypoints[0][left_shoulder_id][:2])
        if right_shoulder_length == 0:
            right_shoulder_length = left_shoulder_length
        if left_shoulder_length == 0:
            left_shoulder_length = right_shoulder_length

        shoulder_length = right_shoulder_length + left_shoulder_length
        keypoints[0][mid_hip_id] = keypoints[0][mid_shoulder_id] + shoulder_length * scale_mid_hip_from_mid_shoulder
        keypoints[0][right_hip_id] = keypoints[0][mid_hip_id] + shoulder_length * scale_right_hip_from_mid_hip
        keypoints[0][left_hip_id] = keypoints[0][mid_hip_id] + shoulder_length * scale_left_hip_from_mid_hip

        pose_data.append(keypoints)

    return pose_data
