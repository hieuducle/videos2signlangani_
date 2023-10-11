import os.path as osp
import cv2
import numpy as np

from video2animation.video_processing import to_mp4


def play_videos(*video_fns,
                output_file='out.mp4', 
                show=False,
                max_resolution=(1080, 1920)):

    cap = [cv2.VideoCapture(video_fn) for video_fn in video_fns if osp.exists(video_fn)]
    
    # Calculate output video resolution
    target_resolution = list(max_resolution) # (0: height, 1: width)
    sum_width_in_same_height = sum([int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * target_resolution[0] / vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) for vid in cap])
    if sum_width_in_same_height <= target_resolution[1]:
        target_resolution[1] = sum_width_in_same_height
    else:
        target_resolution[0] = int(target_resolution[0] * target_resolution[1] / sum_width_in_same_height)
    target_resolution = tuple(target_resolution)

    out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'mp4v'), cap[0].get(cv2.CAP_PROP_FPS), (target_resolution[1], target_resolution[0]))

    frames = [None] * len(cap)
    ret = [None] * len(cap)

    while True:
        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()
        
        if sum(int(ret[i]) for i in range(len(cap))) < len(cap):
            break

        # Combine frames to one single frame
        axis = 1 # 0 = Vertical, 1 = Horizontal
        max_size = 0
        for f in frames:
            max_size = max(max_size, f.shape[1-axis])
        resized_frames = []
        for f in frames:
            scale = max_size / f.shape[1-axis]
            resized_frames.append(_frame_scaling(f, scale))
        combined_frame = np.concatenate(resized_frames, axis=axis)

        # Fit target resolution
        scale = min(target_resolution[0] / combined_frame.shape[0], target_resolution[1] / combined_frame.shape[1])
        combined_frame = _frame_scaling(combined_frame, scale)
        for axis in range(len(target_resolution)):
            shape = list(combined_frame.shape)
            shape[axis] = target_resolution[axis] - shape[axis]
            zeros = np.zeros(shape, dtype=combined_frame.dtype)
            combined_frame = np.concatenate([combined_frame, zeros], axis=axis)

        # Display frame
        out.write(combined_frame)
        if show:
            cv2.imshow('Parallel video', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for c in cap:
        if c is not None:
            c.release()
    out.release()
    to_mp4(output_file)

    cv2.destroyAllWindows()


def _frame_scaling(frame, scale):
    H, W, _ = frame.shape
    size = (int(W*scale), int(H*scale))
    return cv2.resize(frame, size)


if __name__ == '__main__':    
    # play_videos(r'D:\UET\KLTN\vsl\input_data\videos\test-on-full-body.mp4')

    play_videos(r'output_data\videos\test-on-full-body\output_frame_step_5_length1326.mp4',
                r'input_data\openpose-videos\test-on-full-body.avi',
                r'input_data\videos\test-on-full-body.mp4',
                show=False)