"""
git clone https://github.com/ultralytics/yolov5.git để cùng file train.py
"""
from matplotlib import patches as mpatches
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import matplotlib
import torch
from timeit import default_timer as timer
from numba import cuda
from GPUtil import showUtilization as gpu_usage
from tqdm.auto import tqdm
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from pathlib import Path
import copy
import PIL
import easyocr
import splitfolders
import random as rnd
import xml.etree.ElementTree as ET
import glob
import xmltodict
import pandas as pd
import time
import uuid
import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pd.options.mode.chained_assignment = None  # default='warn'


matplotlib.use('TkAgg')


"""
mô tả, chú thích hình ảnh:
    - file: đường dẫn
    - width: rộng
    - height: cao
    - xmin, ymin, xmax, ymax: tọa độ biển số xe
"""
dataset = {
    "file": [],
    "width": [],
    "height": [],
    "xmin": [],
    "ymin": [],
    "xmax": [],
    "ymax": []
}

"""
Mỗi hình ảnh sẽ có một tệp xml mô tả 
"""


img_names=[] 
annotations=[]
for dirname, _, filenames in os.walk("dataset/"):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:]==("png" or "jpg"):
            img_names.append(filename)
        elif os.path.join(dirname, filename)[-3:]=="xml":
            annotations.append(filename)
              
# print(len(img_names), len(annotations))

# print(img_names[:10])

"""
chuyển từ xml -> dictionary
"""

# lấy label của mỗi hình ảnh để train
path_annotations= "dataset/annotations/*.xml" 

for item in glob.glob(path_annotations):
    tree = ET.parse(item)
    
    for elem in tree.iter():
        if 'filename' in elem.tag:
            filename=elem.text
        elif 'width' in elem.tag:
            width=int(elem.text)
        elif 'height' in elem.tag:
            height=int(elem.text)
        elif 'xmin' in elem.tag:
            xmin=int(elem.text)
        elif 'ymin' in elem.tag:
            ymin=int(elem.text)
        elif 'xmax' in elem.tag:
            xmax=int(elem.text)
        elif 'ymax' in elem.tag:
            ymax=int(elem.text)
            
            dataset['file'].append(filename)
            dataset['width'].append(width)
            dataset['height'].append(height)
            dataset['xmin'].append(xmin)
            dataset['ymin'].append(ymin)
            dataset['xmax'].append(xmax)
            dataset['ymax'].append(ymax)
        
classes = ['license']

df=pd.DataFrame(dataset)


# print(df) thông tin file pandas
"""
            file  width  height  xmin  ymin  xmax  ymax
0    Cars204.png    500     375   193   145   279   187
1    Cars329.png    400     255   166   142   224   176
2    Cars350.png    400     268   162   179   211   188
3    Cars320.png    375     500    99   225   257   266
4    Cars225.png    600     375   189   161   384   215
..           ...    ...     ...   ...   ...   ...   ...
466   Cars26.png    400     225   258   184   321   213
467  Cars238.png    600     400   276   186   396   273
468  Cars321.png    301     400   134   235   169   246
469   Cars40.png    400     225   261   186   317   211
470  Cars343.png    400     300   110   186   162   209
"""

def print_random_images(photos: list, n: int = 5, seed=None) -> None:
    if n > 10:
        n=10
    
    if seed:
        rnd.seed(seed)
        
    random_photos = rnd.sample(photos, n)
    
    for image_path in random_photos:
        
        with Image.open(image_path) as fd:
            fig, ax = plt.subplots()
            ax.imshow(fd)           
            ax.axis(False)
            
            for i, file in enumerate(df.file):
                if file in image_path:
                    x1,y1,x2,y2=list(df.iloc[i, -4:])
                        
                    mpatch=mpatches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1, edgecolor='b',facecolor="none",lw=2,)                    
                    ax.add_patch(mpatch)
                    rx, ry = mpatch.get_xy()
                    ax.annotate('licence', (rx, ry-2), color='blue', weight='bold', fontsize=12, ha='left', va='baseline')
                    
photos_path = "dataset/images/*.png"
photos_list = glob.glob(photos_path)

print_random_images(photos_list)


"""
Mô hình YOLO yêu cầu dữ liệu chuẩn hóa (trong phạm vi từ 0 đến 1) 
ở định dạng [class_id, x, y, width, height], trong đó x, y là 
tọa độ giữa hộp giới hạn (với chiều rộng và chiều cao tương ứng). 
Dữ liệu được tính toán phải được lưu dưới dạng tệp .txt với tên tương ứng 
với hình ảnh. Mỗi tệp .txt sẽ trông như thế này:

[class_id, x, y, width, height]

[class_id, x, y, width, height]

[class_id, x, y, width, height]

...
"""


x_pos = []
y_pos = []
frame_width = []
frame_height = []

labels_path = Path("yolov5/dataset/labels")

labels_path.mkdir(parents=True, exist_ok=True)

save_type = 'w'

for i, row in enumerate(df.iloc):
    current_filename = str(row.file[:-4])
    
    width, height, xmin, ymin, xmax, ymax = list(df.iloc[i][-6:])
    
    x=(xmin+xmax)/2/width
    y=(ymin+ymax)/2/height
    width=(xmax-xmin)/width
    height=(ymax-ymin)/height
    
    x_pos.append(x)
    y_pos.append(y)
    frame_width.append(width)
    frame_height.append(height)
    
    txt = '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'
    
    if i > 0:
        previous_filename = str(df.file[i-1][:-4])
        save_type='a+' if current_filename == previous_filename else 'w'
    
    
    with open("yolov5/dataset/labels/" + str(row.file[:-4]) +'.txt', save_type) as f:
        f.write(txt)
        
        
df['x_pos']=x_pos
df['y_pos']=y_pos
df['frame_width']=frame_width
df['frame_height']=frame_height

# print(df)
"""
            file  width  height  ...     y_pos  frame_width  frame_height
0    Cars204.png    500     375  ...  0.442667     0.172000      0.112000
1    Cars329.png    400     255  ...  0.623529     0.145000      0.133333
2    Cars350.png    400     268  ...  0.684701     0.122500      0.033582
3    Cars320.png    375     500  ...  0.491000     0.421333      0.082000
4    Cars225.png    600     375  ...  0.501333     0.325000      0.144000
..           ...    ...     ...  ...       ...          ...           ...
466   Cars26.png    400     225  ...  0.882222     0.157500      0.128889
467  Cars238.png    600     400  ...  0.573750     0.200000      0.217500
468  Cars321.png    301     400  ...  0.601250     0.116279      0.027500
469   Cars40.png    400     225  ...  0.882222     0.140000      0.111111
470  Cars343.png    400     300  ...  0.658333     0.130000      0.076667
"""


"""
Sử dụng thư viện splitfolder, có thể dễ dàng chia hình ảnh và nhãn 
thành các tập hợp training và validation theo định dạng của yolov5
"""

input_folder = Path("dataset")
output_folder = Path("yolov5/dataset/Plate_recognition")
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.8, 0.2),
    group_prefix=None
)
print("Moving files finished.")

def walk_through_dir(dir_path: Path) -> None:
    """Prints dir_path content"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directiories and {len(filenames)} files in '{dirpath}' folder ")

    
walk_through_dir(input_folder)
print()
walk_through_dir(output_folder)


"""
Yolo yêu cầu dữ liệu cấu hình trong tệp .yaml.
"""
import yaml
"""
tạo file plates.ymal để trong thư mục yolov5/data
"""
yaml_file = 'yolov5/data/plates.yaml'
yaml_data = dict(
    path = "dataset/Plate_recognition",
    train = "train",
    val = "val",
    nc = len(classes),
    names = classes
)
with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f, explicit_start = True, default_flow_style = False)

"""

"""
