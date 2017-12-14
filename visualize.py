import os
import cv2
import numpy as np
from skimage.feature import canny
import argparse
from xml.dom import minidom
import shutil

thres = 0
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=4999)
parser.add_argument('--annotation', action='store_true')
parser.add_argument('--gif', action='store_true')
args = parser.parse_args()

def getAnno(filePath):
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

def ifin(res):
    newRes = []
    for i, anno in enumerate(res):
        flag = False
        for j, a in enumerate(res):
            if i == j: continue
            if (anno[0]>a[0] and anno[1]>a[1] and anno[2]<a[2] and anno[3]<a[3]):
                flag = True
                break
        if not flag:
            newRes.append(anno)
    return newRes

def visualize(i):
    img = cv2.imread('/home/dzy/local/traffic_Signs/Img/%05d.jpg'%i)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_1 = np.logical_and(img2[:,:,0]>77, img2[:,:,0]<125)
    mask_2 = np.logical_and(img2[:,:,1]>43, img2[:,:,2]>46)
    mask = np.logical_and(mask_1, mask_2)
    binary_img = np.zeros_like(mask, dtype=np.uint8)
    binary_img[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10,10))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN,kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<thres or h<thres: continue
        roi = img[y:y+h,x:x+w,:]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        tmp = np.zeros_like(roi)
        feature = canny(roi)
        tmp[feature] = 1
        cnt = np.sum(tmp)
        if cnt<100: continue
        res.append((x,y,x+w,y+h))
    newRes = ifin(res)
    for anno in newRes:
        x,y,x2,y2 = anno
        cv2.rectangle(img, (x,y), (x2,y2), (0,255,0), 2)
        # cv2.imshow('roi', tmp)
        # cv2.waitKey(0)
        # im = cv2.resize(img[y:y+h,x:x+w,:], (55,30))
        # feature = hog(im)
        # if clf.predict([feature]) == 0:
            # print "False"
            # continue
        # print 'True'
    if args.annotation:
        try:
            for anno in getAnno('/home/dzy/local/traffic_Signs/Anno/%05d.xml'%i):
                if anno[2]-anno[0]<thres or anno[3]-anno[1]<thres:
                    continue
                cv2.rectangle(img,(anno[0], anno[1]), (anno[2], anno[3]), (255,255,255),2)
        except:
            print 'no annotation file found!'
            return img
    return img

def vis():
    if args.gif:
        os.mkdir('tmp')
        for i in range(args.start, args.end+1):
            im = visualize(i)
            cv2.imwrite('tmp/'+('%05d'%i)+'.jpg', im)
        os.system('convert -loop 0 -delay 100 tmp/*.jpg out.gif')
        os.system('rm -r tmp')
    else:
        i = args.start
        while 1:
            im = visualize(i)
            cv2.imshow('Sign Detction in Wild_DZY', im)
            key = cv2.waitKey(0)
            if key == 1113939:
                i += 1
            elif key == 1113937:
                i-=1
            elif key == 1048603:
                return
            else: continue
            i = min(i, args.end)
            i = max(i, args.start)

def main():
    vis()

if __name__ == '__main__':
    main()