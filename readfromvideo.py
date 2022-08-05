import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
pipeline = keras_ocr.pipeline.Pipeline()
def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    
    #dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
 
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    result = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        masked = cv2.bitwise_and(img, img, mask=mask)
        
                 
    return(masked)

cap = cv2.VideoCapture('image/video_LCD_point2_better.mkv')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out =  cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'X264'), 60, (720,480))
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    ig=inpaint_text(frame, pipeline)
    cv2.imshow('result',ig)
    out.write(ig)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
