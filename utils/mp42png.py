import cv2
import os
cap = cv2.VideoCapture('/media/rit/SSD1TB/datasets/desk.mp4')
savepath = '/media/rit/SSD1TB/datasets/desk'
t1path = os.path.join(savepath,'t1')
if not os.path.isdir(savepath):
    os.mkdir(savepath)
if not os.path.isdir(t1path):
    os.mkdir(t1path)
    import pdb; pdb.set_trace()
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(t1path,'{}.png'.format(i)),frame)
    cv2.imshow('sdaf',frame)
    if i%30==0: print('{}th image saved at {}'.format(i,t1path))
    cv2.waitKey(100)
    i+=1
cap.release()
cv2.destroyAllWindows()