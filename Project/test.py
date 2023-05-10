import numpy as np # linear algebra
import pandas
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from keras.models import Sequential, Model,load_model
import os
import matplotlib.pyplot as plt
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'

img_size=224




model=load_model('keras_resnet50_color_datagen.h5')
#model.summary()
#print(len(model.layers))
i=0
output=pd.DataFrame(columns=['image_name','label'])
for dirname,_,filnames in os.walk('Test'):
    #filnames=random.sample(filnames,10)
    for filename in filnames:
        path = os.path.join(dirname, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img[:, :, [2, 1, 0]]
        test_img = cv2.resize(img, (img_size, img_size))
        test_img = test_img.reshape(1, img_size, img_size, 3)
        prediction = model.predict([test_img])[0]
        max_index = np.argmax(prediction)

        new_row = {'image_name': filename, 'label': max_index}
        new_row=pd.Series(new_row)
        #print(new_row)
        output = output.append(new_row, ignore_index=True)
        #output=pandas.concat(output,new_row,ignore_index=True)
        #print(output)


print(output)
output.to_csv('file1.csv',index=False)


