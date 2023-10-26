import os
import os.path as osp

import env

from text2animation.load_dictionary import load_dictionary
from video2animation.fit_video2character import fit_video2character
from animating.character import load_smplx_3D_model, load_pose
from animating.scene import init_scene
from animating.render import render, init_camera


encoded_word = load_dictionary()


def _set_rest_pose(armature: any, frame: int):
    rest_pose_fn = r'data\rest_pose.pkl'
    load_pose(armature=armature,
              pose_fn=rest_pose_fn,
              frame=frame)


def text2animation(text: str, gender='neutral', frame_step=5, frame_between_word=10, render_video=True):

    # init blender scene
    init_scene()
    armature = load_smplx_3D_model(gender=gender)
    init_camera(armature)

    frame_counter = 0
    _set_rest_pose(armature=armature, frame=frame_counter)
    frame_counter += frame_between_word * 2

    words = text.split(' ')
    for word in words:
        word = word.replace('_', ' ').lower()
        if word not in encoded_word:
            print(f'\"{word}\" not found')
            continue

        keys = encoded_word[word]
        video_path = osp.join(env.INPUT_FOLDER, 'sl_videos', keys+'.mp4')
        character_pose_data, video_path, sl_frame_start, sl_frame_length = fit_video2character(video_path=video_path,
                                                                                               use_hands=True,
                                                                                               use_face=True,
                                                                                               gender=gender,
                                                                                               frame_step=frame_step,
                                                                                               debug=False,
                                                                                               overwrite=False)
        
        # load pose
        pose_files = []
        character_pose_fn = os.listdir(character_pose_data)
        character_pose_fn.sort()
        for pose_fn in character_pose_fn:
            idx = int(osp.splitext(pose_fn)[0])
            if (idx % frame_step == 0 and idx < sl_frame_length - frame_step) or (idx == sl_frame_length - 1):
                pose_files.append((idx, osp.join(character_pose_data, pose_fn)))
        for idx, pose in pose_files:
            load_pose(armature=armature,
                      pose_fn=pose,
                      frame=frame_counter+idx)
        frame_counter += sl_frame_length + frame_between_word

    frame_counter += frame_between_word
    _set_rest_pose(armature=armature, frame=frame_counter)

    if render_video:
        video_name = text + '.mp4'
        output_file = osp.join(env.OUTPUT_VIDEOS, 'sentences', video_name)
        render(0, frame_counter, output_file=output_file)
