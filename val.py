import numpy as np
import cv2

from core import locate_and_correct
from tensorflow import keras
from tensorflow.keras import backend as K

import time

img_w = 320
img_h = 512
n_label = 1+3

def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = int(bbox1[0] - bbox1[2] / 2.0), int(bbox1[1] - bbox1[3] / 2.0)
        xmax1, ymax1 = int(bbox1[0] + bbox1[2] / 2.0), int(bbox1[1] + bbox1[3] / 2.0)
        xmin2, ymin2 = int(bbox2[0] - bbox2[2] / 2.0), int(bbox2[1] - bbox2[3] / 2.0)
        xmax2, ymax2 = int(bbox2[0] + bbox2[2] / 2.0), int(bbox2[1] + bbox2[3] / 2.0)

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1) 
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
 
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

"""-----------------------------------------------定义识别函数-----------------------------------------"""
def recognize(jpg_path, pb_file_path):
    model = keras.models.load_model(pb_file_path,custom_objects={"hard_swish":hard_swish,"relu6":relu6})

    #fobj=open("test_iou_2.txt",'w')
    # 读入图片
    f = open("./val.txt")
    line = f.readline()  
    idx = 0
    while line:
        idx = idx + 1
        img_name = line.split()[0]
        boxlist = img_name.split("-")[2]
        pts = boxlist.split("_")
        pt1 = pts[0].split("&")
        pt2 = pts[1].split("&")
        
        img_path_load = jpg_path + img_name
        #img = cv2.imread(img_path_load)
        data = np.fromfile(img_path_load, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            line = f.readline()
            print("img read false!!!!!!!!!!")
            continue
        #cv2.imshow("src",img)
        sp = img.shape
        img=cv2.resize(img,(img_w,img_h))
        scale_x = img_w/sp[1]
        scale_y = img_h/sp[0]
        boxlabel = [int(pt1[0])*scale_x,int(pt1[1])*scale_y,int(pt2[0])*scale_x,int(pt2[1])*scale_y]
        img_size = img.copy()
        #cv2.imshow("img",img)
        img=img.astype(np.float32)
        # img=np.reshape(img,(1,28,28,1))
        #print("img data type:",img.dtype)

        # 显示图片内容
        start = time.time()
        imgs = []
        imgs.append(img)
        imgs = np.array(imgs)

        img_out_softmax = model.predict(imgs,verbose=2)

        end = time.time()
        print("forward Time: ", end - start)

        #print("img_out_softmax.shape-->", img_out_softmax.shape)
        #res_map = np.squeeze(img_out_softmax)
        for c in range(1,2):
            res_map = img_out_softmax[0,:,:,c]
            #print("res_map.shape-->", res_map.shape)
            
            #print("res_map---max--->",np.max(res_map))
            
            res_map=cv2.resize(res_map,(img_w,img_h))
            #img_mask = res_map.reshape(img_h, img_w, 3)  # 将预测后图片reshape为3维
            img_mask = res_map * 255  # 归一化后乘以255
            
            #img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
            img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型
            #print("img_mask---max--->",np.max(img_mask))
            
            ret,img_mask = cv2.threshold(img_mask,150,255,cv2.THRESH_BINARY)
            #print("img_mask.shape-->", img_mask.shape)
            # 轮廓提取
            contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = img_size.shape[:2]
            index = -1
            max = 0
            maxbox=[]
            for c in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[c])
                if h >=height or w >= width:
                    continue
                area = cv2.contourArea(contours[c])
                if area > max:
                    max = area
                    index = c
                    maxbox = [x, y, x+w, y+h]
            
            if index is not -1:
                cout = contours[index]
                cv2.drawContours(img_size, contours, index, (0, 0, 255), 2, 8)
                cv2.rectangle(img_size, (maxbox[0],maxbox[1]),(maxbox[2],maxbox[3]), (255, 0, 0), thickness=2)
                cv2.imshow("img_size",img_size)
                #print("boxlabel-->",boxlabel)
                #print("maxbox-->",maxbox)
                fiou = iou(boxlabel,maxbox)
                #print("fiou-->",fiou)
                #fobj.write(img_name + " " + str(fiou) + "\n")

        cv2.waitKey(0)
        if idx%100 == 0:
            print("idx-->",idx)

        line = f.readline()

    f.close()



weights_path = "weights.h5"
img_path = "./img/"
#img_path = "F:/BaiduNetdiskDownload/CCPD2019/ccpd_base/"
recognize(img_path, weights_path)