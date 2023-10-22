import os
import os.path as osp
import configargparse

from video2animation.video_processing import to30fps
from video2animation.pose_estimation import estimating_pose


def main(videos_folder: str, div: int, mod: int, debug: bool, overwrite: bool):
    for idx, video in enumerate(os.listdir(videos_folder)):
        try:
            if idx % div == mod:
                video_path = osp.join(videos_folder, video)
                video_path = to30fps(video_path)
                estimating_pose(video_path=video_path,
                                use_hands=True,
                                use_face=True,
                                debug=debug,
                                overwrite=overwrite)
        except KeyboardInterrupt:
            return

# w00863T w01803b w02164b w02328n w03064 w004152

# conda deactivate && source venv/bin/activate
# python building-temp-posedata.py --videos-folder data/input-data/sl_videos --div 5 --mod 0

if __name__ == '__main__':
    args = {}
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      description='Build 3D sign language database from videos',
                                      prog='vsl')
    parser.add_argument('--videos-folder',
                        type=str,
                        required=True,
                        help='Folder contains sign language videos')
    parser.add_argument('--div',
                        type=int,
                        default=1,
                        required=False,
                        help='File id\'s division')
    parser.add_argument('--mod',
                        type=int,
                        default=0,
                        required=False,
                        help='File id\'s modulo')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument('--overwrite',
                        type=bool,
                        default=False,
                        required=False)
    args = vars(parser.parse_args())

    main(**args)