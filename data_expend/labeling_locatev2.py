import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import numpy as np
import cv2, os, sys, getopt
import argparse
import datetime 
import math

Generic_location = os.path.abspath(__file__ + "/..")
print("Generic_location ", Generic_location)

parser = argparse.ArgumentParser()

parser.add_argument('--TrainDIR', type=str, default='', help='path to Trainingdata')
parser.add_argument('--SaveDIR', type=str, default='Label_txt/', help='path to train.txt')
parser.add_argument('--DefaultLabel', type=str, default='0', help='Number of object label')
FLAGS = parser.parse_args()

print("FLAGS.TrainDIR: ", FLAGS.TrainDIR)

day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]

current_time = day + "_" + time + "_"

# Global Variables

# Arguments
pathIMG = ''
pathDIR = ''
pathSAV = ''
regex = ''

pathDIR = FLAGS.TrainDIR
pathSAV = FLAGS.SaveDIR
object_label = FLAGS.DefaultLabel

#===============
bboxes = []
current_label = FLAGS.DefaultLabel
#===============

# Graphical
drawing = False # true if mouse is pressed
cropped = False
ix,iy = -1,-1
rx,ry = -1, -1
angel = -1
bboxes_thickness = 2
bboxes_text_size = 1
bboxes_text_width = 2
# MISC
img_index = 0

def rectangle_color(bbox_class):
    bbox_class = int(bbox_class)
    if bbox_class == 0:
        color = (255, 0, 0)

    elif bbox_class == 1:
        color = (0, 255, 0)

    elif bbox_class == 2:
        color = (0, 0, 255)

    elif bbox_class == 3:
        color = (128, 0, 0)

    elif bbox_class == 4:
        color = (128, 0, 128)

    elif bbox_class == 5:
        color = (32, 154, 0)

    elif bbox_class == 6:
        color = (0, 112, 112)

    elif bbox_class == 7:
        color = (60, 200, 200)

    elif bbox_class == 8:
        color = (10, 160, 255)

    elif bbox_class == 9:
        color = (255, 100, 0)

    else:
        color = (128, 128, 128)

    return color

# Mouse callback function
def draw(event,x,y,flags,param):
    global ix, iy, rx, ry, drawing, img, DEFAULT, cropped , pathIMG , angel

    cropped = False
    
    img = DEFAULT.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if bboxes != None:
                for i in range(len(bboxes)):
                    #cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]),bboxes_thickness)
                    cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]),bboxes_thickness)
                    tmp_x=bboxes[i][2]-bboxes[i][0]
                    tmp_y=bboxes[i][3]-bboxes[i][1]
                    cv2.line(img,(bboxes[i][0]-tmp_x,bboxes[i][1]-tmp_y),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]),bboxes_thickness)
                    cv2.circle(img, (bboxes[i][0],bboxes[i][1]), 5, (0,0,255),3)
                    cv2.rectangle(img,(bboxes[i][6],bboxes[i][7]),(bboxes[i][8],bboxes[i][9]), rectangle_color(bboxes[i][4]),bboxes_thickness)
            # show rectangle when pull
            #cv2.rectangle(img,(ix,iy),(x,y), rectangle_color(current_label), bboxes_thickness)
            cv2.line(img,(ix,iy),(x,y), rectangle_color(current_label), bboxes_thickness)
            cv2.circle(img,(ix,iy), 5, (0,0,255),3)
            tmp_x=x-ix
            tmp_y=y-iy
            cv2.line(img,(ix-tmp_x,iy-tmp_y),(x,y), rectangle_color(current_label), bboxes_thickness)

            # cv2.putText(img, str(current_label), (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, \
            #     rectangle_color(current_label), bboxes_text_width, cv2.LINE_AA)
            rx, ry = x, y
            ###calculate angel
            angel=math.atan2((ry-iy),(rx-ix))*-1
            #angel=math.atan((bboxes[i][3]-bboxes[i][1])/(bboxes[i][2]-bboxes[i][0]))
            angel=round(angel/math.pi*180)
            cv2.imshow(pathIMG, img)

        else:
            if bboxes != None:
                for i in range(len(bboxes)):
                    #
                    #cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]), bboxes_thickness)
                    cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]), bboxes_thickness)
                    tmp_x=bboxes[i][2]-bboxes[i][0]
                    tmp_y=bboxes[i][3]-bboxes[i][1]
                    cv2.line(img,(bboxes[i][0]-tmp_x,bboxes[i][1]-tmp_y),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][4]),bboxes_thickness)
                    cv2.circle(img, (bboxes[i][0],bboxes[i][1]), 5, (0,0,255),3)
                    cv2.rectangle(img,(bboxes[i][6],bboxes[i][7]),(bboxes[i][8],bboxes[i][9]), rectangle_color(bboxes[i][4]),bboxes_thickness)
                    #################
                    #cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][4]), bboxes_text_width, cv2.LINE_AA)
            #cv2.putText(img, str(current_label), (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(current_label), bboxes_text_width, cv2.LINE_AA)

            cv2.line(img,(0,y),(img.shape[1], y),(0,255,255),1)
            cv2.line(img,(x,0),(x, img.shape[0]),(0,255,0),1)
            cv2.circle(img, (x, y), 5, (0,0,255),3)

        # pass

    elif event == cv2.EVENT_LBUTTONUP:
        rx, ry = x, y
        target_x, target_y, end_x, end_y, target_angle=ix, iy ,rx ,ry ,angel

        l = 0
        while (1):
            l = cv2.waitKey(1)
            tmp_x=rx-target_x
            tmp_x=target_x-tmp_x
            tmp_y=ry-target_y
            tmp_y=target_y-tmp_y
            cv2.rectangle(img,(tmp_x,tmp_y),(rx,ry),rectangle_color(current_label),bboxes_thickness)
            cv2.line(img,(target_x,target_y),(end_x,end_y), rectangle_color(current_label), bboxes_thickness)
            tmp_x1=end_x-target_x
            tmp_y2=end_y-target_y
            cv2.line(img,(target_x-tmp_x1,target_y-tmp_y2),(end_x,end_y), (140, 140, 255), bboxes_thickness)
            cv2.imshow(pathIMG, img)
            if (l == ord('b')):
                bounding_ix,bounding_iy=tmp_x,tmp_y
                bounding_rx,bounding_ry=rx,ry
                if bounding_ix > bounding_rx :
                        bounding_ix, bounding_rx = bounding_rx, bounding_ix
                if bounding_iy > bounding_ry :
                        bounding_iy, bounding_ry = bounding_ry, bounding_iy
                break

        # if ix > rx :
        #     ix, rx = rx, ix
        # if iy > ry :
        #     iy, ry = ry , iy

        bboxes.append([target_x, target_y, end_x, end_y, current_label,target_angle,bounding_ix,bounding_iy,bounding_rx,bounding_ry])
        #bboxes.append([ix, iy, rx, ry, current_label,angel])
        drawing = False
        
        print("bboxes ", bboxes)
        # for i in range(len(bboxes)):
        #     cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),rectangle_color(bboxes[i][4]),bboxes_thickness)
        #     cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][4]), bboxes_text_width, cv2.LINE_AA)

        cv2.imshow(pathIMG, img)
        print('end')



# Crop Image
def crop(ix,iy,x,y):
    global img, DEFAULT, cropped

    img = DEFAULT.copy()
    cv2.imshow(pathIMG,img)

    if (abs(ix - x) < abs(iy -y)):
        img = img[iy:y, ix:x]
    else:
        img = img[iy:y, ix:x]

def WriteRectangle():
    global ix, iy, rx, ry, pathIMG, bboxes
   
    line = ""
    
    for i in range(len(bboxes)):
        # line = line + ',' + str(bboxes[i][0]) + ',' + str(bboxes[i][1]) + ',' + str(bboxes[i][2]) + 
        #     ',' + str(bboxes[i][3])
        line = line + ',' + str(bboxes[i][0]) + ',' + str(bboxes[i][1]) + ',' + str(bboxes[i][5])+','\
            +str(bboxes[i][6]) + ',' + str(bboxes[i][7])+',' + str(bboxes[i][8]) + ',' + str(bboxes[i][9])
    line = str(Generic_location + "/" + pathIMG) + line  + '\n'

    if not os.path.exists(pathSAV):
        os.makedirs(pathSAV)

    with open(pathSAV + 'train_locate' + "_" + current_time + '.csv', 'a') as f:
        f.writelines(line)

# Main Loop
def loop():
    global img, pathIMG, current_label

    cv2.namedWindow(pathIMG, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(pathIMG, draw)

    while (1):
        cv2.imshow(pathIMG, img)
        k = cv2.waitKey(1) & 0xFF

        if (k == 27):
            bboxes.clear()
            print("Cancelled Crop")
            break

        elif (k == ord('0')):
            current_label = 0
            pass

        elif (k == ord('1')):
            current_label = 1
            pass

        elif (k == ord('2')):
            current_label = 2
            pass

        elif (k == ord('3')):
            current_label = 3
            pass

        elif (k == ord('4')):
            current_label = 4
            pass

        elif (k == ord('5')):
            current_label = 5
            pass

        elif (k == ord('6')):
            current_label = 6
            pass

        elif (k == ord('7')):
            current_label = 7
            pass

        elif (k == ord('8')):
            current_label = 8
            pass

        elif (k == ord('9')):
            current_label = 9
            pass

        elif (k == ord('c')):
            img = DEFAULT.copy()

            if len(bboxes) != 0:
                bboxes.pop()

            for i in range(len(bboxes)):
                cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),rectangle_color(bboxes[i][4]),bboxes_thickness)
                cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][4]), bboxes_text_width, cv2.LINE_AA)

            cv2.imshow(pathIMG, img)

            print("bboxes ", bboxes)

            pass

        elif (k == ord('s')):# and cropped):
            WriteRectangle()
            print("======================================\n")
            print("                 ||                   ")
            print("                 ||                   ")
            print("                \||/                   ")
            print("                 \/                   \n")
            print("================Saved=================")
            print("Data : ", pathIMG)
            print("Label : ", bboxes)
            print("Number of Label :", len(bboxes))
            print("======================================\n")

            bboxes.clear()
            #save(img)
            break

    # print("Done!")
    cv2.destroyAllWindows()

# Iterate through images in path
def getIMG(path):
    global img, DEFAULT, pathIMG
    directory = os.fsencode(path)
    for filename in os.listdir(directory):
        # Get Image Path
        pathIMG = path + filename.decode("utf-8")
        print("======================================")
        print("Current Data:", pathIMG)

        # Read Image
        img = cv2.imread(pathIMG,-1)
        
        DEFAULT = img.copy()

        # Draw image
        loop()

    return 0

# Main Function
def main():
    global img, DEFAULT

    if (pathDIR != ''):
        # Print Path
        print("pathDIR: " + pathDIR)

        # Cycle through files
        getIMG(pathDIR)

    elif (pathIMG != ''):
        # Print Path
        print("img: " + pathIMG)

        # Load Image
        img = cv2.imread(pathIMG,-1)
        DEFAULT = img.copy()

        # Draw Image
        loop()

# Run Main
if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
