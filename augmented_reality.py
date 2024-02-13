import cv2
import numpy as np


min_matches=15
img_rgb=cv2.imread('E:\Augmented Reality\Quaid_Mazar.PNG')
cap=cv2.imread('E:\Augmented Reality\Quaid_Mazar.PNG',0)
model=cv2.imread('E:\Augmented Reality\competition.png',0)

#initiate ORB detector
orb =cv2.ORB_create()


#create brute force matcher object
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Compute model keypoint and its descriptors
kp_model, des_model = orb.detectAndCompute(model,None)

#Compute scene keypoint and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap,None)

#Match frame descriptors with model descriptors
matches=bf.match(des_model, des_frame)

matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > min_matches:
    #draw first 15 matches
    cap = cv2.drawMatches(model, kp_model, cap,kp_frame,
                          matches[:min_matches], 0, flags=2)


    cv2.imshow('keypoint', cap)
    cv2.waitKey(0)

else:
    print("Not enough matches -%d/%d" % (len(matches),min_matches))

#assuming matches stores the matches found and returned
#differenciate b/w source points and destination points

src_pts=np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
dst_pts=np.float32([kp_model[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#Draw rectangle
h,w = model.shape
pts=np.float32([[0,0],[0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

#project corners into frame
dst = cv2.perspectiveTransform(pts, M)

#connect them with lines

img2 = cv2.polylines(cap, [np.int32(dst)],True,255,3,cv2.LINE_AA)
cv2.imshow('frame',cap)
cv2.waitKey(0)




















    
