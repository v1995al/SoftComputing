import cv2
import numpy as np
from vektor import distance, pnt2line2
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import time
from scipy import ndimage

def traziLinije(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('',  gray)
    edges = cv2.Canny(gray, 50, 100, apertureSize = 3)
    cv2.imshow('', edges)

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
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data() #niz slika
    
    novi_train_images = []
    novi_train_labels = []

    print('Preiprema trening slika')
    for i in range(len(train_images)):
        ret, thresh = cv2.threshold(train_images[i], 127, 255, cv2.THRESH_BINARY)
        img, contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 1:
            [x,y,w,h] = cv2.boundingRect(contours[0])
            cropImg = train_images[i][y:y+h+1, x:x+w+1]
            cropImg = cv2.resize(cropImg, (28,28), interpolation = cv2.INTER_AREA)
            novi_train_images.append(cropImg)
            novi_train_labels.append(train_labels[i])

    novi_test_images = []
    novi_test_labels = []
    print('Preiprema test slika')
    for i in range(len(test_images)):
        ret, thresh = cv2.threshold(test_images[i], 127, 255, cv2.THRESH_BINARY)
        img, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 1:
            [x,y,w,h] = cv2.boundingRect(contours[0])
            cropImg = test_images[i][y:y+h+1, x:x+w+1]
            cropImg = cv2.resize(cropImg, (28,28), interpolation = cv2.INTER_AREA)
            novi_test_images.append(cropImg)
            novi_test_labels.append(test_labels[i])

    novi_train_images = np.array(novi_train_images,'float32')
    novi_test_images = np.array(novi_test_images,'float32')

    dim_data = np.prod(novi_train_images.shape[1:]) #mnozi sve u nizu
    train_data = novi_train_images.reshape(novi_train_images.shape[0], dim_data)
    test_data = novi_test_images.reshape(novi_test_images.shape[0], dim_data)
 
    nClasses = len(np.unique(novi_train_labels))
 
    train_data /= 255.0
    test_data /= 255.0
 
    train_labels_one = to_categorical(novi_train_labels)
    test_labels_one = to_categorical(novi_test_labels)
 
    #relu = kada je x manje od 0 sve je nula a x vece od 0 onda je fja x
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape = (dim_data,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
 
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) #pravi model
 
    model.fit(train_data, train_labels_one, batch_size = 256, epochs = 20, verbose = 1, validation_data = (test_data, test_labels_one)) #trenira
   
    [test_loss, test_acc] = model.evaluate(test_data, test_labels_one)
 
    print('Ishod testiranja: = greska {}, tacno = {}'.format(test_loss, test_acc))
 
    model.save('model.h5')

def deskew(img):
    m = cv2.moments(img)

    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def klasifikujBrojeve(img):

    global model

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = deskew(img)
    
    img = cv2.dilate(img, (4, 4))

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    imgThresh, contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    konturePovrsine = []

    for j in range(len(contours)):
        konturePovrsine.append(cv2.contourArea(contours[j]))

    najvecaKontura = np.argmax(konturePovrsine)

    [x,y,w,h] = cv2.boundingRect(contours[najvecaKontura])
    cropImg = img[y:y+h+1, x:x+w+1]
    cropImg = cv2.resize(cropImg, (28,28), interpolation = cv2.INTER_AREA)

    # if len(contours) == 1:
    #     [x,y,w,h] = cv2.boundingRect(contours[0])
    #     cropImg = img[y:y+h+1, x:x+w+1]
    #     cropImg = cv2.resize(cropImg, (28,28), interpolation = cv2.INTER_AREA)
    # else:
    #     print("ujebo")
    #     cropImg = img
    #     cropImg = cv2.resize(cropImg, (28,28), interpolation = cv2.INTER_AREA)

    cv2.imshow('', cropImg)

    predictNum = cropImg.flatten() / 255.0 # flatten pretvara matricu u vektor
    #jenda slika i pretvramo je u niz od jednog clana
    predictNum = np.array([predictNum], 'float32')

    prediction = model.predict(predictNum)[0] #prime niz slika za predvideti, i vracaa niz od jednog niza [1 0 0 0 0 0 000 0 0 0]

    return np.argmax(prediction) #vraca max index tog niza

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

elements = []
t = 0
suma = 0

def pratiObjekat(img, linije):

    global elements
    global t
    global suma
    granice = [
    ([230, 230, 230], [255, 255, 255])]

    start_time = time.time()

    kernel = np.ones((2,2),np.uint8) #matrica 2x2 jedinica
    donjaGranica = np.array(granice[0][0])
    gornjaGranica = np.array(granice[0][1])
    
    donjaGranica = np.array(donjaGranica, dtype = "uint8")
    gornjaGranica = np.array(gornjaGranica, dtype = "uint8")
    mask = cv2.inRange(img, donjaGranica, gornjaGranica) #vraca samo boje koju u granicama

    img0 = 1.0*mask

    img0 = cv2.dilate(img0,kernel) # uveca bele piksele, kao da bolduje
    img0 = cv2.dilate(img0,kernel)

    #cv2.imshow('aa', img0)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    for i in range(nr_objects):
        loc = objects[i] #pozicija obj
        (xc,yc) = ((loc[1].stop + loc[1].start)/2, #centar obj
                   (loc[0].stop + loc[0].start)/2)
        (dxc,dyc) = ((loc[1].stop - loc[1].start), #duzina i sirina obj
                   (loc[0].stop - loc[0].start))

        xc = int(xc)
        yc = int(yc)

        if(dxc>10 or dyc>10): #dovoljno velik obj
            cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1) #crtaj tanki krug
            elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t} #klasa 
            # find in range
            lst = inRange(20, elem, elements) #elements list iz predhodnog frejma, vraca listu elemenata koji su u obegu manjem od 20 od trenutnog elem
            nn = len(lst)
            if nn == 0: #ne postoji
                elem['id'] = nextId()
                elem['t'] = t
                elem['pass0'] = False #prosao ili nije liniju
                elem['pass1'] = False
                elem['broj'] = None #funkcija koja klasifikuje
                elements.append(elem)
            elif nn == 1: #ako postoji
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                        
    for el in elements: #svi elementi 
        tt = t - el['t']
        if(tt<3):
            x1, y1, x2, y2 = linije[0]
            dist, pnt, r = pnt2line2(el['center'], (x1, y1), (x2,y2)) #racuna udaljenost od linije i IF

            if el['broj'] is None:
                xc, yc = el['center']
                el['broj'] = klasifikujBrojeve(img[yc-12:yc+12, xc-12:xc+12])

            if r>0: #ne znamo
                cv2.line(img, pnt, el['center'], (0, 255, 25), 1) #linija od obj ka linijji
                c = (25, 25, 255)
                if(dist<9): #ako je uddaljenost jako mala
                    c = (0, 255, 160) 
                    if el['pass0'] == False:
                        el['pass0'] = True
                        suma += el['broj']
                        

                cv2.circle(img, el['center'], 16, c, 2)

            x1, y1, x2, y2 = linije[1]
            dist, pnt, r = pnt2line2(el['center'], (x1, y1), (x2,y2)) #racuna udaljenost od linije i IF
            if r>0: #ne znamo
                cv2.line(img, pnt, el['center'], (0, 255, 25), 1) #linija od obj ka linijji
                c = (25, 25, 255)
                if(dist<9): #ako je uddaljenost jako mala
                    c = (0, 255, 160) 
                    if el['pass1'] == False:
                        el['pass1'] = True
                        suma -= el['broj']
                cv2.circle(img, el['center'], 16, c, 2)

            id = el['id']
            cv2.putText(img, str(el['broj']), 
                (el['center'][0]+10, el['center'][1]+10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    cv2.putText(img, 'Suma: ' + str(suma), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)    
    t += 1

def main(brVideo):
    global niz
    global model
    global suma
    global elements
    global t
    cap = cv2.VideoCapture('video-'+str(brVideo)+'.avi')
    flag, img = cap.read()

    linije = traziLinije(img)

    model = load_model('model.h5')

    while(cap.isOpened()): 
        flag, img = cap.read()
        if flag == True:

            pratiObjekat(img, linije)

            for i in range(0, len(linije)):       
                cv2.line(img, (linije[i][0], linije[i][1]), (linije[i][2], linije[i][3]), [0,i*100,255], 3)
                cv2.putText(img,text=str(i),org=(linije[i][0], linije[i][1]), fontFace = 0, fontScale = 0.7, color=(0,0,255))

            cv2.imshow('Video'+str(brVideo),img)
            
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
        else:
            break

    niz.append(suma)
    print('Suma ' + str(suma))
    suma = 0
    elements = []
    t = 0

    cap.release()
    cv2.destroyAllWindows()

#treniranje()
niz = []
model = []
for i in range(0,10):
    main(i)

file = open('out.txt', 'w')

upis = 'Valentina \nfile\tsum\n'

for i in range(len(niz)):
    upis += 'video-'+str(i)+'.avi\t'+str(niz[i])+'\n'

file.write(upis)
file.close

