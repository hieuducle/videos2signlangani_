import os
import os.path as osp
import cv2

import env

from video2animation.video_processing import to30fps
from video2animation.pose_estimation import estimating_pose
from video2animation.sl_detector import get_sl_range
from video2animation.joints_autofill import fill_joints
from video2animation.fit_pose2character import fit_pose2character


def fit_video2character(video_path,
                        use_hands=True,
                        use_face=True,
                        gender='neutral',
                        frame_step=1,
                        debug=False,
                        overwrite=False):
    """
    Return:
        :character_pose_data_folder: the folder contain character pose data
        :30fps_video: the video used in pose estimation (with is necessary for future comparison)
        :sl_frame_start: index of the frame which starting hand signing from the 30fps video
        :sl_frame_length: the number of frames which contain hand sign
    """

    print(f'\n Generating 3D pose for {gender} character from {video_path}')

    video_path = to30fps(video_path)
    pose_folder = estimating_pose(video_path, use_hands=use_hands, use_face=use_face, debug=debug, overwrite=overwrite)
    cap = cv2.VideoCapture(video_path)
    sl_frame_range = get_sl_range(openpose_save=pose_folder,
                                  fps=cap.get(cv2.CAP_PROP_FPS),
                                  frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                  frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_data = fill_joints(pose_folder=pose_folder,
                            use_hands=use_hands,
                            use_face=use_face)
    character_pose_data_folder = osp.join(env.OUTPUT_3D_POSE, osp.splitext(osp.split(video_path)[1])[0], gender)
    os.makedirs(character_pose_data_folder, exist_ok=True)
    for idx, keypoints in enumerate(pose_data[sl_frame_range[0]:sum(sl_frame_range)]):
        if idx % frame_step != 0:
            continue

        print(f'\nWorking with {video_path} on frame {idx+sl_frame_range[0]}')
        character_pose_fn = osp.join(character_pose_data_folder, str(idx).zfill(12)+'.pkl')
        fit_pose2character(keypoints=keypoints,
                           result_fn=character_pose_fn,
                           gender=gender,
                           use_hands=use_hands,
                           use_face=use_face,
                           overwrite=overwrite)
    return character_pose_data_folder, video_path, sl_frame_range[0], sl_frame_range[1]
