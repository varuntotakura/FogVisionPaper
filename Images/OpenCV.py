import cv2
import imutils

#image = cv2.imread("google-alphabet.jpg")
image = cv2.imread("../Images/cap3.jpg")
cv2.imshow("Image",image)
copy = image.copy()
grey = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grey",grey)
blur = cv2.GaussianBlur(copy,(11,11),0)
cv2.imshow("Blur",blur)
edge1 = cv2.Canny(blur,50,60)
#cv2.imshow("Edge1",edge1)
edge2 = cv2.Canny(copy,100,100)
#cv2.imshow("Edge2",edge2)
rect = cv2.rectangle(copy, (290, 10), (385, 160), (0, 0, 255), 2)
#cv2.imshow("Rectangle",rect)
cir = cv2.circle(rect, (760, 15), 15, (150, 0, 0), -1)
#cv2.imshow("Circle", cir)
line = cv2.line(cir, (cir.shape[1], 0), (0, cir.shape[0]), (255, 255, 255), 5)
#cv2.imshow("Line", line)
text = cv2.putText(cir, "VARUN", (cir.shape[1]-100, cir.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#cv2.imshow("Text", text)
thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
copy = image.copy()
for c in cnts:
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    cv2.drawContours(copy, [c], -1, (240, 0, 159), 2)
# draw the total number of contours found in purple
tex = "I found {} objects!".format(len(cnts))
cv2.putText(copy, tex, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
#cv2.imshow("Contours", copy)
mask = cv2.erode(thresh, None, iterations=5)
#cv2.imshow("Eroded", mask)
mask = cv2.dilate(mask, None, iterations=5)
#cv2.imshow("Dilated", mask)
bit = cv2.bitwise_and(image, image, mask=mask)
#cv2.imshow("Output", bit)
edge = cv2.Canny(bit,900,150)
cv2.imshow("Edge",edge)
cv2.waitKey(0)
