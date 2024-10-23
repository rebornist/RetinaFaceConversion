# RetinaFace Extension

해당 기능은 [Pytorch_Retinaface](https://https://github.com/biubug6/Pytorch_Retinaface)에 구현된 기능을 분석하고, 이를 기반으로 추가 기능을 개발하기 위한 저장소입니다. 

## Installation
##### 라이브러리 설치

```Shell
pip install torch torchvision
pip install retinaface-pytorch opencv-python datasets
```

##### GitHub 저장소 클론

```Shell
git clone https://github.com/biubug6/Pytorch_Retinaface.git
mv Pytorch_Retinaface core
```

##### Data Download
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. 디렉토리 구조
```Shell
  ./dataset/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt 에는 val/images/ 폴더 내 파일명만 입력해야합니다.

## Training

1. 학습 시작 전 ``core/data/config.py train.py``에서 network 옵션(예: batch_size, min_sizes, steps 등)을 확인해야 합니다.


2. WIDER FACE 기반 모델 학습 커멘드:
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python3 train.py --network resnet50 or mobile0.25
  ```

## Testing

1. 테스트 전 ``core/data/config.py test_widerface.py``에서 옵션을 확인해야 합니다.

2. WIDER FACE 기반 모델 테스트 커멘드:
  ```Shell
  python3 test_widerface.py --trained_model ./core/weights/mobilenet0.25_Final.pth --network mobile0.25 --dataset_folder ./dataset/widerface/val/images/ --save_folder ./results/mobilenet/ -s
  ```