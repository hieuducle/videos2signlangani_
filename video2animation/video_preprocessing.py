import os
import os.path as osp
import sys

import ffmpeg


# Add ffmpeg binary to current process PATH
for folder in os.listdir('.local/'):
    if folder.startswith('ffmpeg') and osp.exists(osp.join('.local/', folder,'bin')):
        sys.path.append(osp.join('.local/', folder,'bin'))


def change_fps(src, target_fps) -> [str, ]:
    output_file = osp.splitext(src)[0] + f'fps={target_fps}' + osp.splitext(src)[1]

    stream = ffmpeg.input(src)
    stream = stream.filter('fps', fps=target_fps)
    stream = ffmpeg.output(stream, output_file)

    return output_file, stream