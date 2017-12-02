# extract annotation from xml
# from xml.dom import minidom
# file = '/home/dzy/local/traffic_Signs/Anno/00001.xml'
# e = minidom.parse(file)
# objList = e.getElementsByTagName('object')

# print len(objList)
# a =  objList[0].getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0]
# print (a.firstChild.data)
def getAnno(filePath):
    from xml.dom import minidom
    def getVal(x, tag):
        return x.getElementsByTagName(tag)[0].firstChild.data
    objs = minidom.parse(filePath).getElementsByTagName('object')
    objectList = []
    for o in objs:
        tmp = []
        coords = o.getElementsByTagName('bndbox')[0]
        for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
            tmp.append(int(getVal(coords, tag)))
        objectList.append(tmp)
    return objectList


import cv2
import numpy as np
import selectivesearch
import time

img = cv2.imread('/home/dzy/local/traffic_Signs/Img/00227.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_1 = np.logical_and(img2[:,:,0]>77, img2[:,:,0]<125)
mask_2 = np.logical_and(img2[:,:,1]>43, img2[:,:,2]>46)
mask = np.logical_and(mask_1, mask_2)
binary_img = np.zeros_like(mask, dtype=np.uint8)
binary_img[mask] = 255
cv2.imshow('123', binary_img)
cv2.waitKey(0)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))
# opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN,kernel)
# closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10,10))
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN,kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
cv2.imshow('123', closed)
cv2.waitKey(0)
imgRes = img.copy()
mask2 = np.logical_not(closed)
imgRes[mask2] = (0,0,0)
contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w<50 and h<50: continue
    cv2.rectangle(imgRes, (x,y), (x+w,y+h), (0,255,0),2)
print len(contours)
cv2.drawContours(imgRes,contours[1],-1,(0,0,255),3)  
cnt = contours[14]
epsilon = 0.1*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.imshow('123', imgRes)
cv2.waitKey(0)
# img_lbl, regions = selectivesearch.selective_search(imgRes, scale=5, sigma=0.9, min_size=100)
# candidates = set()
# for r in regions:
#     if r['rect'] in candidates:
#         continue
#     # if r['size'] < 500:
#     #     continue
#     candidates.add(r['rect'])
# for c in candidates:
#     cv2.rectangle(imgRes, (c[0],c[0]+c[2]), (c[1],c[1]+c[3]), (255,255,255))
# cv2.imshow('123', imgRes)
# cv2.waitKey(0)
# objList = getAnno('/home/dzy/local/traffic_Signs/Anno/00001.xml')
# start = time.time()
# img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
# candidates = set()
# for r in regions:
#     if r['rect'] in candidates:
#         continue
#     if r['size'] < 500:
#         continue
#     candidates.add(r['rect'])
# for c in candidates:
#     cv2.rectangle(img, (c[0],c[0]+c[2]), (c[1],c[1]+c[3]), (255,255,255))
# cv2.imshow('123', img)
# cv2.waitKey(0)
# print 'select search:', str(time.time()-start), 's'
# print 'find ', len(candidates), 'of', len(regions)

# obj = objList[3]
# cv2.rectangle(img, (obj[0],obj[1]), (obj[2],obj[3]), (255,255,255))
# # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('123', img)
# cv2.waitKey(0)

# whiteSet = []
# ide = []
# cnt = 0
# def connect(x,y):
#     if abs(x[0]-y[0])==1:
#         return True
#     if abs(x[1]-y[1])==1:
#         return True
#     return False

# for i in range(closed.shape[0]):
#     for j in range(closed.shape[1]):
#         if closed[i,j]:
#             whiteSet.append((i,j))
#             ide.append(cnt)
#             cnt+=1
