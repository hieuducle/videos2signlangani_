import os
import os.path as osp

import env


def estimating_pose(video_path: str,
                    use_hands=True,
                    use_face=True,
                    debug=False,
                    overwrite=False):
    
    video_path = osp.abspath(video_path)
    vid_name = osp.splitext(osp.split(video_path)[1])[0]
    model_folder = env.OPENPOSE_MODEL
    output_folder = osp.join(env.OPENPOSE_OUTPUT_KEYPOINT, vid_name)
    os.makedirs(output_folder, exist_ok=True)
    cmd = f'OpenPoseDemo --video \"{video_path}\" --write_json \"{output_folder}\" --model_folder \"{model_folder}\" --display 0 --number_people_max 1 --model_pose BODY_25'
    if use_hands:
        cmd += ' --hand'
    if use_face:
        cmd += ' --face'
    if debug:
        openpose_images_folder = osp.join(env.OPENPOSE_OUTPUT_IMAGES, vid_name)
        cmd += f' --write_images \"{openpose_images_folder}\"'
    else:
        cmd += f' --render_pose 0'
    if not overwrite:
        saved_keypoints = os.listdir(output_folder)
        frame_start = 0
        while (vid_name+'_'+str(frame_start).zfill(12)+'_keypoints.json') in saved_keypoints:
            frame_start += 1
        cmd += f' --frame_first {frame_start}'
    print(cmd, flush=True)
    os.system(cmd)
    print('\n')
    
    return output_folder