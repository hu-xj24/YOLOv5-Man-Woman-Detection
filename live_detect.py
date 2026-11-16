
import cv2
import torch
import numpy as np
import time

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

# ==================== 参数配置 ====================

WEIGHTS = "best_4090.pt"
IMG_SIZE = 640
CONF_THRES = 0.5
IOU_THRES = 0.45

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== 加载模型 ====================

model = attempt_load(WEIGHTS, map_location=device)
model.float()
model.eval()

names = model.names
stride = int(model.stride.max())

print("Device:", device)
print("Classes:", names)

colors = {
    0: (255, 0, 0),
    1: (0, 0, 255)
}

# ==================== 打开摄像头 ====================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window = "YOLOv5 Live Detection"

cv2.namedWindow(window)

# 创建置信度调节滑块
cv2.createTrackbar(
    "Confidence",
    window,
    int(CONF_THRES * 100),
    100,
    lambda x: None
)

# ==================== 实时推理 ====================

try:
    with torch.inference_mode():
        while True:

            start = time.perf_counter()

            ret, frame = cap.read()

            if not ret:
                break

            # 镜像显示
            frame = cv2.flip(frame, 1)

            # 实时获取置信度阈值
            conf_thres = max(
                0.01,
                cv2.getTrackbarPos("Confidence", window) / 100
            )

            # 1. 图像预处理
            img = letterbox(
                frame,
                IMG_SIZE,
                stride=stride
            )[0]

            # BGR -> RGB, HWC -> CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            # NumPy -> Tensor
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0

            # CHW -> BCHW
            img = img.unsqueeze(0)

            # 2. 神经网络前向传播
            pred = model(img)[0]

            # 3. 置信度筛选与非极大值抑制
            detections = non_max_suppression(
                pred,
                conf_thres,
                IOU_THRES
            )[0]

            # 4. 绘制检测框
            if len(detections):

                detections[:, :4] = scale_coords(
                    img.shape[2:],
                    detections[:, :4],
                    frame.shape
                ).round()

                for x1, y1, x2, y2, conf, cls in detections.cpu().tolist():

                    cls = int(cls)

                    color = colors[cls]
                    label = f"{names[cls]} {conf:.2f}"

                    x1, y1 = int(x1), int(y1)
                    x2, y2 = int(x2), int(y2)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

            # 5. 计算处理帧率
            fps = 1 / (time.perf_counter() - start)

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  Threshold: {conf_thres:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 6. 实时显示
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
