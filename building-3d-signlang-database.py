import os
import os.path as osp
import configargparse

from video2animation.fit_video2character import fit_video2character


def main(videos_folder: str, gender: str, frame_step: int, debug: bool, overwrite: bool):
    for video in os.listdir(videos_folder):
        video_path = osp.join(videos_folder, video)
        fit_video2character(video_path=video_path,
                            use_hands=True,
                            use_face=True,
                            gender=gender,
                            frame_step=frame_step,
                            debug=debug,
                            overwrite=overwrite)


if __name__ == '__main__':
    args = {}
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      description='Build 3D sign language database from videos',
                                      prog='vsl')
    parser.add_argument('--videos-folder',
                        type=str,
                        required=True,
                        help='Folder contains sign language videos')
    parser.add_argument('--frame-step',
                        type=int,
                        default=1,
                        required=False,
                        help='Frame step')
    parser.add_argument('--gender',
                        type=str,
                        default='neutral',
                        required=False,
                        help='Gender: male, female or neutral')
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