#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('pip install easyocr')
get_ipython().system('pip install torch torchvision torchaudio')


# In[21]:


import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[29]:


IMAGE_PATH = r'C:\Users\logi\Downloads\demo jpeg text.png'


# In[23]:


reader = easyocr.Reader(['en'])
result = reader.readtext(IMAGE_PATH)
result


# In[24]:


top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX


# In[25]:


img = cv2.imread(IMAGE_PATH)
img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
img = cv2.putText(img,text,top_left, font, 0.5,(255,255,255),2,cv2.LINE_AA)
plt.imshow(img)
plt.show()


# In[28]:


img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    img = cv2.putText(img,text,(20,spacer), font, 1.5,(0,255,0),2,cv2.LINE_AA)
    spacer+=15
    
plt.imshow(img)
plt.show()


# In[30]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')


# In[31]:


import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


IMAGE_PATH = r'C:\Users\logi\Downloads\demo jpeg text.png'


# In[33]:


def recognize_text(img_path):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)


# In[34]:


result = recognize_text(IMAGE_PATH)


# In[35]:


result


# In[36]:


img_1 = cv2.imread(IMAGE_PATH)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
plt.imshow(img_1)


# In[37]:


def overlay_ocr_text(img_path, save_name):
   img = cv2.imread(img_path)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   dpi = 80
   fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
   plt.figure()
   f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
   axarr[0].imshow(img)
   
   # recognize text
   result = recognize_text(img_path)

   # if OCR prob is over 0.5, overlay bounding box and text
   for (bbox, text, prob) in result:
       if prob >= 0.5:
           # display 
           print(f'Detected text: {text} (Probability: {prob:.2f})')

           # get top-left and bottom-right bbox vertices
           (top_left, top_right, bottom_right, bottom_left) = bbox
           top_left = (int(top_left[0]), int(top_left[1]))
           bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

           # create a rectangle for bbox display
           cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

           # put recognized text
           cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)
       
   # show and save image
   axarr[1].imshow(img)
   plt.savefig(f'./output/{save_name}_overlay.jpg', bbox_inches='tight')


# In[38]:


overlay_ocr_text(IMAGE_PATH, '2_handwriting')


# In[39]:


overlay_ocr_text(IMAGE_PATH, '3_digits')


# In[40]:


get_ipython().system('pip install pyttsx3')


# In[42]:


result = recognize_text(IMAGE_PATH)

sentence = ''
for (bbox, text, prob) in result:
    sentence += f'{text} '
print(sentence)


# In[43]:


import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 100)
engine.say(sentence)
engine.runAndWait()


# In[ ]:




