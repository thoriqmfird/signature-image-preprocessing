import numpy as np
import cv2

image = cv2.imread('signature.jpg')

nomer = 299
widthImg = 1170                                                                      
heightImg = 1998                                                                     

image = cv2.resize(image, (widthImg,heightImg))
orig = image.copy()
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5,5),0)
edge_image = cv2.Canny(blurred_image,0,50)
orig_edge = edge_image.copy()
(contours,_) = cv2.findContours(edge_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype= np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)
    if len(approx) == 4:
        target = approx
        break

approx = rectify(target)
pts2 = np.float32([[0,0],[widthImg,0],[widthImg,heightImg],[0,heightImg]])

M = cv2.getPerspectiveTransform(approx,pts2)
dst = cv2.warpPerspective(orig,M,(widthImg,heightImg))

cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
dstAdative = cv2.adaptiveThreshold(dst, 255, 1, 1, 23, 7) #11, 4 23, 7
dstAdative = cv2.bitwise_not(dstAdative)
dstAdative = cv2.medianBlur(dstAdative,3)
cv2.imshow("test",dstAdative)
# cv2.imwrite(f"BW-A-.png",dstAdative)

final_grey = dst[5:(heightImg-5), 5:(widthImg-5)]
final_grey = cv2.resize(final_grey, (widthImg,heightImg))
final_img = dstAdative[5:(heightImg-5), 5:(widthImg-5)]
final_img = cv2.resize(final_img, (widthImg,heightImg))
# cv2.imshow("Hc",final_img)

widthCropped = int(widthImg/3)
heightCropped = int(heightImg/6)
countx = 1

for r in range(0,final_img.shape[0],heightCropped): #Crop BW Image
    for c in range(0,final_img.shape[1],widthCropped):
        cv2.imwrite(f"BW-A-{nomer}-{countx}.png",final_img[r+7:r+(heightCropped), c:c+widthCropped])
        countx = countx+1
        if countx == 18:
            continue
county = 1
for s in range(0,final_grey.shape[0],heightCropped): #Crop Grey Image
    for t in range(0,final_grey.shape[1],widthCropped):
        cv2.imwrite(f"Grey-A-{nomer}-{county}.png",final_grey[s:s+heightCropped, t:t+widthCropped])
        county = county+1

cv2.waitKey(0)
