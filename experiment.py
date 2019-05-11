import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

#%%
img = cv2.imread('./train/anpr_ocr/train/img/A001CB06.png',0)
img1 = cv2.imread('./train/anpr_ocr/train/img/A001CB06.png',0)
plt.imshow(img,cmap='gray')
plt.show()
img = 255-img
img[img<130] = 0
img[img>=130] = 255
plt.imshow(img,cmap='gray')
plt.show()
print(img.shape)

#%%
def Hist_col(img):
    width = img.shape[1]
    height = img.shape[0]
    w_cal = np.zeros((width))
    for col in range(width):
        for row in range(height):
            if img[row,col] > 0:
                w_cal[col] += 1
    w_cal[w_cal>30] = 0
    w_cal[w_cal<7] = 0
    
    return w_cal
w_cal = Hist_col(img)

def Hist_row(img):
    h_cal = np.zeros((height))
    for row in range(height):
        for col in range(width):
            if img[row,col] > 10:
                h_cal[row] += 1
            

#%%
col_loc = np.zeros(12)

n = 0
for i in range(len(w_cal)):
    try:
        if(w_cal[i]==0 and w_cal[i+1]>0):
            col_loc[n] = i-2
            n+=1
        if(w_cal[i]>0 and w_cal[i+1]==0):
            col_loc[n] = i+3
            n+=1
    except:
        continue
col_loc = col_loc.reshape(-1,2)
print(col_loc)

row_loc = np.zeros(12)

#print(col_loc.shape)
for i in range(col_loc.shape[0]):
    patch = np.zeros((height,int(col_loc[i][1]-col_loc[i][0])))
    patch = img[7:31,int(col_loc[i][0]):int(col_loc[i][1])]
    plt.imshow(patch,cmap='gray')
    plt.show()
    patch = cv2.resize(patch,(20,20))
    plt.imshow(patch,cmap='gray')
    plt.show()
plt.imshow(img,cmap='gray')
print(img.shape)











