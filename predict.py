import torch
import easyocr
import numpy as np
import cv2 
import matplotlib.pyplot as plt


## load model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='MODEL/weights/best.pt', force_reload=True)


## chọn ngôn ngữ của biển số xe
reader = easyocr.Reader(['en'])


def get_plates_xy(frame: np.ndarray, labels: list, row: list, width: int, height: int, reader: easyocr.Reader) -> tuple:
    '''Lấy kết quả từ easyOCR cho mỗi frame(hình ảnh), trả lại tọa độ bounding box '''

    x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]
                                                                * width), int(row[3]*height)  # BBOx coordniates
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox
    # , paragraph="True", min_size=50)
    ocr_result = reader.readtext(np.asarray(
        plate_crop), allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    return ocr_result, x1, y1


def detect_text(i: int, row: list, x1: int, y1: int, ocr_result: list, detections: list, yolo_detection_prob: float = 0.3) -> list:

    if row[4] >= yolo_detection_prob:  # discard predictions below the value
        if (len(ocr_result)) > 0:
            for item in ocr_result:
                detections[i][0] = item[1]
                detections[i][1] = [x1, y1]
                detections[i][2] = item[2]

    return detections


def is_adjacent(coord1: list, coord2: list) -> bool:
    MAX_PIXELS_DIFF = 50

    if (abs(coord1[0] - coord2[0]) <= MAX_PIXELS_DIFF) and (abs(coord1[1] - coord2[1]) <= MAX_PIXELS_DIFF):
        return True
    else:
        return False


def sort_detections(detections: list, plates_data: list) -> list:
    for m in range(0, len(detections)):
        for n in range(0, len(plates_data)):
            if not detections[m][1] == [0, 0] and not plates_data[n][1] == [0, 0]:
                if is_adjacent(detections[m][1], plates_data[n][1]):
                    if m != n:
                        temp = detections[m]
                        detections[m] = detections[n]
                        detections[n] = temp

    return detections


def delete_old_labels(detections: list, count_empty_labels: list, plates_data: list, frames_to_reset: int = 3) -> tuple:
    for m in range(0, len(detections)):
        if detections[m][0] == 'None' and not count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] += 1
        elif count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] = 0
            plates_data[m] = ['None', [0, 0], 0]
        else:
            count_empty_labels[m] = 0

    return plates_data, count_empty_labels


def overwrite_plates_data(detections: list, plates_data: list, plate_lenght=None) -> list:
    if (detections[i][2] > plates_data[i][2] or detections[i][2] == 0):
        if plate_lenght:
            if len(detections[i][0]) == plate_lenght:
                plates_data[i][0] = detections[i][0]
                plates_data[i][2] = detections[i][2]
        else:
            plates_data[i][0] = detections[i][0]
            plates_data[i][2] = detections[i][2]
    plates_data[i][1] = detections[i][1]

    return plates_data

## đường dẫn ảnh
test_photo_path = "source/" + "test1.jpg"

results = model(test_photo_path)
detections=np.squeeze(results.render())

labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
image = cv2.imread(test_photo_path)
width, height = image.shape[1], image.shape[0]

print(f'Ảnh Rộng, Cao: {width},{height}. Số lượng biển số xe: {len(labels)}')

for i in range(len(labels)):
    row = coordinates[i]
    if row[4] >= 0.6:
        x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
        plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
        ocr_result = reader.readtext((plate_crop), paragraph="True", min_size=120, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text=ocr_result[0][1]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6) ## BBox
        cv2.putText(image, f"{text}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        plt.axis(False)
        plt.imshow((image)[...,::-1])
        
        print(f'Dự đoán: {i+1}. Trọng số: {row[4]:.2f}, easyOCR Kết quả: {ocr_result}')
        
        
"""

"""
