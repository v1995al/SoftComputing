import cv2
import numpy as np
from vektor import distance
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense


def traziLinije(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize = 3)

    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength = 190, maxLineGap = 20)

    list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        list.append([x1, y1, x2, y2])

    list2 = []
    for i in range(0, len(list)):
        if list[i] not in list2:
            temp = True
            for j in range(0, len(list2)):
                if abs(list[i][0]-list2[j][0]) < 20 or abs(list[i][1]-list2[j][1]) < 7 or abs(list[i][2]-list2[j][2]) < 7 or abs(list[i][3]-list2[j][3]) < 20:
                    temp = False
            if temp == True:
                list2.append(list[i])

    retList = []
    x1, y1, x2, y2 = list2[0]
    srednjaTacka1 = ((x1+x2)/2, (y1+y2)/2)
    retList.append(list2[0]) # 0 -ADD
    x1, y1, x2, y2 = list2[1]
    srednjaTacka2 = ((x1+x2)/2, (y1+y2)/2)
    retList.append(list2[1]) # 1 -SUB

    if distance((0,500), srednjaTacka1) < distance((0,500), srednjaTacka2):
        retList[0] = list2[1]
        retList[1] = list2[0]

    return retList

def treniranje():
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
    dim_data = np.prod(train_images.shape[1:])
    train_data = train_images.reshape(train_images.shape[0], dim_data)
    test_data = test_images.reshape(test_images.shape[0], dim_data)
 
    nClasses = len(np.unique(train_labels))
 
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
 
    train_labels_one = to_categorical(train_labels)
    test_labels_one = to_categorical(test_labels)
 
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape = (dim_data,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
 
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
    model.fit(train_data, train_labels_one, batch_size = 256, epochs = 20, verbose = 1, validation_data = (test_data, test_labels_one))
   
    [test_loss, test_acc] = model.evaluate(test_data, test_labels_one)
 
    print('Evaulation result on Test Data : Loss = {}, accuracy = {}'.format(test_loss, test_acc))
 
    model.save('model.h5')
 
    return model

#treniranje()

cap = cv2.VideoCapture('video-0.avi')
flag, img = cap.read()

linije = traziLinije(img)
while(cap.isOpened()): 
    flag, img = cap.read()
    if flag == True:

        for i in range(0, len(linije)):       
            cv2.line(img, (linije[i][0], linije[i][1]), (linije[i][2], linije[i][3]), [0,i*100,255], 3)
            cv2.putText(img,text=str(i),org=(linije[i][0], linije[i][1]), fontFace = 0, fontScale = 0.7, color=(0,0,255))

        cv2.imshow('Video',img)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    else:
        break
      
cap.release()
cv2.destroyAllWindows()


