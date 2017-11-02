###1.importing libraries

import cv2
import matplotlib.pyplot as plt

####2.load our image
#note1-open cv always loads image in BGR format
ana_image=cv2.imread('3.jpg')
######3. convert image into greyscale
#note2. open cv face detector expects image in grayscale format
grey_ana_image=cv2.cvtColor(ana_image,cv2.COLOR_BGR2GRAY)

###4. SHOW IMAGE
#cmap->color map
plt.imshow(grey_ana_image,cmap='gray')

####5. train with file haarcascade_frontalface_alt.xml  
ana_trained=cv2.CascadeClassifier('/home/anand/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

####6. detect face using function detectMultiScale()
#detectMultiScale() :this function detects face and returns co-ordinate of faces
ana_detectedface=ana_trained.detectMultiScale(grey_ana_image,scaleFactor=1.2)

##cut


#go over list of faces and draw them as rectangles on original colored 
def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for (x, y, w, h) in ana_detectedface :     
         cv2.rectangle(ana_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
plt.imshow(convertToRGB(ana_image))
