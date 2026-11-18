import cv2, glob
from src.vnplate.config import load_config
from src.vnplate.detector import load_detector, detect_plates

cfg = load_config()
model = load_detector(cfg.detector)

# Bỏ [:50:5] nếu bạn muốn kiểm tra toàn bộ tập val
paths = sorted(glob.glob("data/dataset_resplit/images/val/*.png"))

for p in paths:
    img = cv2.imread(p)
    dets = detect_plates(img, model, cfg.detector)
    
    # Chỉ xử lý và in ra những ảnh có đúng 2 biển số
    if len(dets) == 2:
        print(p.split("/")[-1], len(dets), [round(d.conf, 2) for d in dets])
        for d in dets:
            cv2.polylines(img, [d.corners.astype(int)], True, (0, 255, 0), 2)
        cv2.imwrite("outputs/stage1_" + p.split("/")[-1].replace(".png", ".jpg"), img)
