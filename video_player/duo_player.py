import numpy as np
import cv2

def duo_player(first_video, second_video):
    cap = [cv2.VideoCapture(i) for i in [first_video, second_video]]
    out = None
    target_resolution = None

    frames = [None] * 2
    ret = [None] * 2

    while True:
        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()

        if out is None:
            # 0: Width
            # 1: Height
            target_resolution = (sum([f.shape[1] for f in frames]), max([f.shape[0] for f in frames]))
            out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'mp4v'), cap[0].get(cv2.CAP_PROP_FPS), target_resolution)

        combine_frame = np.zeros(shape=(target_resolution[1], target_resolution[0], 3), dtype=np.uint8)

        sum_col = 0
        for i,f in enumerate(frames):
            if ret[i] is True:
                combine_frame[:f.shape[0], sum_col:sum_col+f.shape[1]] = f
                sum_col += f.shape[1]

        cv2.imshow('Duo player', combine_frame)
        out.write(combine_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if sum([int(r is True) for r in ret]) == 0:
            break

    for c in cap:
        if c is not None:
            c.release()
    
    if out is not None:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    duo_player(r'D:\UET\KLTN\vsl\output_data\videos\Easy On Me - Adele -Contemporary Dance Sabrina Ryckaert_ver0.1\output_frame_step_30_length200.mp4',
               r'D:\UET\KLTN\vsl\output_data\videos\Easy On Me - Adele -Contemporary Dance Sabrina Ryckaert_ver0.2\output_frame_step_30_length200.mp4')