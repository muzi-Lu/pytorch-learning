
import cv2

def main():
    vc = cv2.VideoCapture("/home/benben/code/pytorch-learning/data_preprocess/MOV_1237.mp4")
    c = 1

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    timeF = 3

    while rval:
        rval, frame = vc.read()
        if(c % timeF == 0):
            cv2.imwrite('/home/benben/code/pytorch-learning/data_preprocess/img_data/' + str(c+4500) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)

    vc.release()

if __name__ == '__main__':
    main()