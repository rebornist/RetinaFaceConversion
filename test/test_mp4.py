import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from core.data.config import cfg_mnet
from core.layers.functions.prior_box import PriorBox
from core.models.retinaface import RetinaFace
from core.utils.box_utils import decode, decode_landm
from core.utils.nms.py_cpu_nms import py_cpu_nms
from core.utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device):
    print('Loading pretrained model from {}'.format(pretrained_path))

    
    # 만약 .pth 파일을 로드할 경우 (일반적인 PyTorch 모델 가중치)
    if pretrained_path.endswith('.pth'):
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    
    # 만약 .pt 파일을 로드할 경우 (TorchScript 모델)
    elif pretrained_path.endswith('.pt'):
        model = torch.jit.load(pretrained_path, map_location=device)
        print('Loaded TorchScript model.')

    else:
        raise ValueError("Unsupported model file extension. Use .pth for PyTorch models or .pt for TorchScript models.")
    
    return model


# ONNX 모델 로드
pretrained_path = "./weights/new_mobilenet0.25_Final.pth"
cfg = cfg_mnet
# RetinaFace 모델 로드 (모델 정의에 따라 조정이 필요)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, pretrained_path, device)
net.eval()
print('Finished loading model!')
print(net)
cudnn.benchmark = True
net = net.to(device)

# MP4 비디오 열기
video_path = "~/dev/TorchCppExample/input_video.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장을 위한 설정 (선택 사항)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output_with_faces.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

_t = {'forward_pass': Timer(), 'misc': Timer()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    img = np.float32(frame)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()

    # 후처리 함수를 사용하여 바운딩 박스 및 랜드마크 추출
    for b in dets:
        if b[4] < 0.6:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(frame, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # 랜드마크 그리기
        cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
        
    # 결과 프레임을 비디오 파일로 저장
    out.write(frame)

# 자원 해제
cap.release()
out.release()