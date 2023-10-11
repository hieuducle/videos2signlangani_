# Hướng dẫn sử dụng

## Cài đặt

### Requirement

- Python 3.10
- [FFMPEG](https://ffmpeg.org/download.html)
- [OPENPOSE DEMO](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0)
- CUDA
- PyTorch 1.x.x
- Các thư viện trong [requirement.txt](requirements.txt)

**Chú ý: FFMPEG và OPENPOSE cần được link vào PATH**

### Config thư mục

Các đường dẫn được định nghĩa tại file `env.py`

### Tải các model

Tạo thư mục `models` dùng để chứa các model sẽ sử dụng

#### Openpose

[Download](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models) các file trong thư mục trên vào `models\openpose`

Sau đó chạy file `getModels`

#### Sign language detection

[Download](https://github.com/sign-language-processing/detection-train/tree/master/models/openpose-body/py) file `model.h5` vào thư mục `models\sl_detection`

#### SMPLX

Truy cập [website](https://smpl-x.is.tue.mpg.de/) và đăng ký tài khoản. Sau khi đăng nhập, truy cập các link sau để download và giải nén vào thư mục `models`:

- [smplx](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip)
- [vposer](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=V02_05.zip)

Sau đó tải file [config](https://github.com/vchoutas/smplify-x/blob/master/cfg_files/fit_smplx.yaml) vào thư mục `data\input-data`

## Sử dụng

### Xây dựng cơ sở dữ liệu ngôn ngữ ký hiệu 3D

```sh
python building-3d-signlang-database.py --videos-folder {path tới thư mục chứa các video}
```

Ngoài ra còn có các tham số khác, `-h` để biết thêm chi tiết

Chú ý: Để tối ưu thời gian chạy, có thể chạy nhiều tiến trình `building-3d-signlang-database.py` cùng lúc trên các thư mục khác nhau. Đoạn mã trong `folder_splitter.py` có thể hỗ trợ chia một thư mục thành nhiều phần. Cách dùng:

```sh
python folder_splitter.py --folder {path tới thư mục cần chia} --div {số thư mục con được tạo ra}
```
