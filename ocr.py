#import numpy as np
import cv2
import numpy as np
import pandas as pd
#from sklearn.datasets import load_breast_cancer
import msvcrt
import keyboard
import sklearn

# Load an color image in grayscale
img = cv2.imread('test.jpg',0)
#imS = cv2.resize(img, (960, 540))                       # Resize image
#cv2.imshow("output", imS)                              # Show image
#cv2.waitKey(0)                                         # Display the image infinitely until any keypress
#import numpy.lib.stride_tricks.as_strided

A = img

def test():
    cv2.waitKey(0)
    key = keyboard.read_key()
    key = keyboard.read_key()
    print(key.name)
    return key


def saveforNN(A,action,file):
    height=A.shape[0]
    width=A.shape[1]

    ndWindow=A[i:(i+height),0:width].reshape(1,height*width)
    window=np.concatenate((np.matrix([Action,height,width]),ndWindow),axis=1)
    
    with open(file, 'a') as file:
        file.writelines(window)   

def Hastext(A):
    cv2.imshow('Windows',A)
    cv2.waitKey(0)
    key = keyboard.read_key()
    print(key)

    switcher={'space':0,
              'enter':1,
              '-':2,
              '+':3}

    Action=swithcher.get(key.name,0)

    saveforNN(A,action,'Hastext.txt')
       
    return Action

def Identifychar(A):
    imS = cv2.resize(A, (960, 540))
    cv2.imshow('position',ims)
    cv2.waitKey(0)
    key = keyboard.read_key()
    print(key.name)

    saveforNN(A,key.name,'Identifychar.txt')

    return key.name

def charpos(window):

    height=int(window.shape[0])
    width=int(height*0.75)
    
    all_chars=[]

    for j in range(0,A.shape[1] - width + 1,width):
        char=Identifychar(window[0:height,j:j+width])
        all_chars.append([c,j]) 

    return all_chars            

#xsize=int(A.shape[0]/100)
#ysize=int(A.shape[1]/2)


def textwindows(A,height):

    width=int(A.shape[1])
    step=int(height/2) 

    if A.shape[0] < height:
        return
    
    for i in range(0,A.shape[0] - height + 1,step):      
        window=A[i:(i+height),0:width]
        Action=Hastext(window)
        
        if  Action==1:
                Char_Positions=charpos(window)
                
        elif Action==2:
                windows=textwindow(A[i:i+height*2,0:width],height*2)
                
        elif Action==3:
                windows=textwindow(window,height/2)
    
    return ndWindows         



def buildData():
    import os
    ndWindows=pd.DataFrame()  

    for fontsize in [18,84]:
        for x in os.listdir('.'):
            if x.endswith('.jpg'):
                img = cv2.imread(x,0)
                lines=int(img.shape[0]/fontsize)
                #lines=80
                ndWindows.append(pd.DataFrame(textwindows(img,fontsize)),ignore_index=True)
        ndWindows.to_csv(('test'+fontsize+'.csv'))



def trainWindows(file):
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.externals import joblib
    
    ndWindows=pd.read_csv(file)

    loaded_model = joblib.load('textwindows.sav')

    mlp = MLPClassifier(hidden_layer_sizes=(25,), max_iter=20, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=100,
                    learning_rate_init=.001, learning_rate='invscaling',
                    activation='logistic')
    
    X, y=ndWindows.take(range(1,ndWindows.shape[1]-1),axis=1),ndWindows['0']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Testing set score: %f" % mlp.score(X_test, y_test))

    joblib.dump(mlp, 'textwindows.sav')

   
    #KNN=KNeighborsClassifier(n_neighbors=5)
    #KNN.fit(X_train, y_train)
    #print("Training set score: %f" % KNN.score(X_train, y_train))
    #print("Testing set score: %f" % KNN.score(X_test, y_test))
    #results= pd.DataFrame((mlp.predict(X), KNN.predict(X),y))

        
    return mlp     


def showwindow(window):
    cv2.imshow("output", window[0])
    
      


#print(A)
#as_strided=np.lib.stride_tricks.as_strided
#all_windows = as_strided(img,(int((A.shape[0] - xsize + 1) / xstep), int((A.shape[1] - ysize + 1) / ystep),
                          # xsize, ysize),(A.strides[0] * xstep, A.strides[1] * ystep, A.strides[0], A.strides[1]))

#cv2.imshow('d',img)

