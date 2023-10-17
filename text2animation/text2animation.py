import os
import os.path as osp

import env

from load_dictionary import load_dictionary
from video2animation.fit_video2character import fit_video2character
from animating.character import load_smplx_3D_model, load_pose
from animating.scene import init_scene
from animating.render import render, init_camera


encoded_word = load_dictionary()


def _set_default_pose(armature, frame):
    pass


def text2animation(text: str, gender='neutral', frame_step=5, frame_between_word=10):

    # init blender scene
    init_scene()
    armature = load_smplx_3D_model(gender=gender)
    init_camera(armature)

    frame_counter = 0
    words = text.split(' ')
    for word in words:
        word = word.replace('_', ' ').lower()
        print(word)
        if word not in encoded_word:
            # print()
            continue
        keys = encoded_word[word]
        video_path = osp.join(env.INPUT_FOLDER, 'videos', keys+'.mp4')
        print(video_path)
        continue
        character_pose_data, video_path, sl_frame_start, sl_frame_length = fit_video2character(video_path=video_path,
                                                                                               use_hands=True,
                                                                                               use_face=True,
                                                                                               gender=gender,
                                                                                               frame_step=frame_step,
                                                                                               debug=False,
                                                                                               overwrite=False)
        
        # load pose
        pose_files = []
        for pose_fn in os.listdir(character_pose_data):
            if int(osp.splitext(pose_fn)[0]) % frame_step == 0 and int(osp.splitext(pose_fn)[0]) < sl_frame_length:
                pose_files.append(osp.join(character_pose_data, pose_fn))
        for pose_fn in pose_files:
            load_pose(armature=armature,
                      pose_fn=pose_fn,
                      frame=frame_counter)
            frame_counter += frame_step
        frame_counter += frame_between_word - frame_step
