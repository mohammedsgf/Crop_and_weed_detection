import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def getWeeds(net, frame, confi_thresh, NMSthresh, labelsPath, show=False):

    """
    This function takes a network, frame, confidance threshold, NMS threshold, labels path and show permsion
    and return the high confidance and low confidance weeds in the frame
    
    """

    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    
    #determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    (H, W) = frame.shape[:2]

    #makeing blob image for the net input
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 512),swapRB=True, crop=False)

    #set the net input
    net.setInput(blob)

    #forward the input and get the output
    outlayers = net.forward(ln)

    #true boxes with high confidance
    boxes = []
    confidances = []
    classIDs = []

    #true boxes with high confidance
    unkown_boxes = []
    unkown_confidances = []
    unkown_classIDs = []

    weeds = []
    true_weeds = []

    #iterate over the output layers
    for out in outlayers:

        #itrate over the output grid vector
        for grid in out:
            
            #get the confidance of the two classes 0:Crops and 1:Weeds
            #The Yolov3 grid vector contians |x|y|w|h|Pc|class1 Predection|class2 Predection|....etc 
            #Taking the last two items on the vector gives us the predection of out classes
            scores = grid[5:]

            #get the indicaes of the max predection
            predicted_class = np.argmax(scores)

            #get the predection
            predect = scores[predicted_class]

            #predect if there is an object
            object_pred = grid[5]

            #To just predect weeds
            if predicted_class == 1:

                #rescale the x,y,w and h to the image scale
                box = grid[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                #use the center (x, y)-coordinates to derive the top and
                #and left corner of the bounding box
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                #update our list of bounding box coordinates, confidences,
                #and class IDs
                boxes.append([x, y, int(width), int(height),centerX ,centerY])
                confidances.append(float(predect))
                classIDs.append(predicted_class)

                

    #apply non-maxima suppression to suppress weak, overlapping bounding
    #boxes                                          
    true_boxes = cv2.dnn.NMSBoxes(boxes, confidances, 0, NMSthresh) #TODO: Play with confidance parmater to have the best results
    

    if show:
        if len(true_boxes) > 0:
            #loop over the indexes we are keeping
            for i in true_boxes.flatten():

                #extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                (centerX , centerY) = (boxes[i][4], boxes[i][5])
                cv2.circle(frame, (centerX,centerY) , 5, (0, 0, 255) , 4)

                if confidances[i] > confi_thresh:
                    true_weeds.append([x, y, int(width), int(height),centerX ,centerY,confidances[i]])
                else:
                    weeds.append([x, y, int(width), int(height),centerX ,centerY,confidances[i]])

                #draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidances[i])
                cv2.putText(frame, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        # if len(unkown_objects) > 0:
        #     #loop over the indexes we are keeping
        #     for i in unkown_objects.flatten():
        #         #extract the bounding box coordinates
        #         (x, y) = (unkown_boxes[i][0], unkown_boxes[i][1])
        #         (w, h) = (unkown_boxes[i][2], unkown_boxes[i][3])
        #         (centerX , centerY) = (unkown_boxes[i][4], unkown_boxes[i][5])
        #         weeds.append([x, y, int(width), int(height),centerX ,centerY,unkown_confidances[i]])
        #         cv2.circle(frame, (centerX,centerY) , 5, (255, 0, 0) , 4)
        #         #draw a bounding box rectangle and label on the image
        #         color = [int(c) for c in COLORS[unkown_classIDs[i]]]
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #         text = "{}: {:.4f}".format(LABELS[unkown_classIDs[i]], unkown_confidances[i])
        #         cv2.putText(frame, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


        det = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12,8))
        plt.imshow(det)
        plt.show()

    return true_weeds, weeds


if __name__ == "__main__":
    #load the class labels our YOLO model was trained on
    labelsPath = 'C:\\Users\\MohammedSGF\\Desktop\\Senior Project\\WeedDetec\\Crop_and_weed_detection\\performing_detection\\data\\names\\obj.names'



    #load weights and cfg
    weightsPath = 'C:\\Users\\MohammedSGF\\Desktop\\Senior Project\\WeedDetec\\Crop_and_weed_detection\\performing_detection\\data\\weights\\crop_weed_detection.weights'
    configPath = 'C:\\Users\\MohammedSGF\\Desktop\\Senior Project\\WeedDetec\\Crop_and_weed_detection\\performing_detection\\data\\cfg\\crop_weed.cfg'


    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    frame = cv2.imread('C:\\Users\\MohammedSGF\\Desktop\\Senior Project\\WeedDetec\\Crop_and_weed_detection\\performing_detection\\data\\images\\test_4.jpg')
    
    #parameters
    confi_thresh = 0.3
    NMSthresh = 0.5

    true_weeds, weeds = getWeeds(net, frame, confi_thresh, NMSthresh, labelsPath, show=True)
    print(true_weeds)
    print(weeds)