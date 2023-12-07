import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt


image_dir = 'Image_Classifiation\\cropped'
messi = os.listdir('Image_Classifiation\\cropped\\lionel_messi')
maria = os.listdir('Image_Classifiation\\cropped\\maria_sharapova')
roger = os.listdir('Image_Classifiation\\cropped\\roger_federer')
serena = os.listdir('Image_Classifiation\\cropped\\serena_williams')
virat = os.listdir('Image_Classifiation\\cropped\\virat_kohli')

print("--------------------------------------\n")

print('The number of images of Lionel Messi is',len(messi))
print('The number of images of Maria Sharapova is',len(maria))
print('The number of images of Roger Federer is',len(roger))
print('The number of images of Serena Williams is',len(serena))
print('The number of images of Virat Kohli is',len(virat))
print("--------------------------------------\n")


dataset=[]
label=[]
img_siz=(128,128)
IMAGE_SIZE=128

input_shape = ( 128,128, 3)

for i , image_name in tqdm(enumerate(messi),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
for i , image_name in tqdm(enumerate(maria),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(serena),desc="Serena William"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(virat),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)

dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")

x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255
print("--------------------------------------\n")


print("--------------------------------------\n")
'''data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
 '''

model=tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])
print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

y_train_one_hot = to_categorical(y_train,num_classes=5)
y_test_one_hot = to_categorical(y_test,num_classes=5)

print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train_one_hot,epochs=50,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test_one_hot)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print('classification Report\n',classification_report(y_test_one_hot,y_pred))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")

def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

# List of paths to multiple images you want to predict
image_paths_to_predict = [
     'Image_Classifiation\\cropped\\lionel_messi\\lionel_messi1.png',
     'Image_Classifiation\\cropped\\maria_sharapova\\maria_sharapova1.png',
     'Image_Classifiation\\cropped\\roger_federer\\roger_federer1.png',
     'Image_Classifiation\\cropped\\serena_williams\\serena_williams2.png',
     'Image_Classifiation\\cropped\\virat_kohli\\virat_kohli1.png'
    # Add more image paths as needed
]

# Preprocess and predict for each image
for image_path in image_paths_to_predict:
    single_image = preprocess_single_image(image_path)
    single_image = np.expand_dims(single_image, axis=0)
    predictions = model.predict(single_image)
    predicted_class = np.argmax(predictions)
    
    class_names = ['Lionel Messi', 'Maria Sharapova', 'Roger Federer', 'Serena williams', 'Virat kohli']
    predicted_label = class_names[predicted_class]

    print(f" Predicted image: {predicted_label}")




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print('------------------------------------')

print(model.evaluate(x_train))
print(model.evaluate(y_test_one_hot))


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(50), acc, label='Training Accuracy')
plt.plot(range(50), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(50), loss, label='Training Loss')
plt.plot(range(50), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(r'C:\Users\USER\Desktop\Models\Image_Classifiation\image_accuracy_plot.png')
   