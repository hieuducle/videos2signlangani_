# To run with blender --python. Not required for other python
import bpy
import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
# To run with blender --python. Not required for other python

import os
import os.path as osp
import configargparse

from pose2animation.pose2animation import pose2animation
from pose2animation.animating.character import load_smplx_3D_model
from pose2animation.animating.scene import init_scene
from pose2animation.animating.render import render, init_camera
from video_player.play_videos import play_videos

def main(file_name=str,
         frame_step=int,
         frame_end=int,
         gender=str):
    
    file_name = 'test-on-full-body'
    frame_step = 5
    frame_end = 1326
    gender = 'female'
    
    # file_name = 'D0001B'
    # frame_step = 5
    # frame_end = 129
    # gender = 'neutral'

    # file_name = 'Easy On Me - Adele -Contemporary Dance Sabrina Ryckaert'
    # frame_step = 30
    # frame_end = 200
    # gender = 'female'

    keypoints_folder = f'input_data/keypoints/{file_name}'

    # init blender scene
    init_scene()
    armature = load_smplx_3D_model(gender=gender)
    init_camera(armature)

    frame_counter = 0
    while frame_counter < frame_end:
        if len(os.listdir(path=keypoints_folder)) <= frame_counter:
            print(f'Waiting the {frame_counter}^th keypoints file')
            continue
        file = os.listdir(path=keypoints_folder)[frame_counter]
        pose2animation(armature=armature,
                       frame=frame_counter,
                       keypoints_fn=osp.join(keypoints_folder,file),
                       use_hands=True,
                       use_face=True,
                       gender=gender)
        frame_counter += frame_step
    
    output_video_fn = f'output_data/videos/{file_name}/{gender}_frame_step_{frame_step}_length{frame_end}.mp4'

    render(frame_start=0, frame_end=frame_end, width=1500, height=2000, output_file=output_video_fn, fps=30)
    
    play_videos(output_video_fn,
                f'input_data/openpose-videos/{file_name}.mp4',
                f'input_data/videos/{file_name}.mp4',
                output_file=osp.splitext(output_video_fn)[0]+'.parallel'+osp.splitext(output_video_fn)[1])


if __name__ == '__main__':
    args = {}
    # parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    #                                  description='Pose to animation',
    #                                  prog='pose2animation')
    # parser.add_argument('--file-name',
    #                    type=str,
    #                    required=True,
    #                    help='The name of keypoints\' folder and source video')
    # parser.add_argument('--frame-step',
    #                    type=int,
    #                    default=1,
    #                    required=False,
    #                    help='Frame step')
    # parser.add_argument('--frame-end',
    #                    type=int,
    #                    required=True,
    #                    help='Frame end')
    # parser.add_argument('--gender',
    #                    type=str,
    #                    default=neutral,
    #                    required=False,
    #                    help='Gender: male, female or neutral')
    # args = vars(parser.parse_args())
    main(**args)