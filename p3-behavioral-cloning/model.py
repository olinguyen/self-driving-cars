import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from keras.layers.core import Dense, Activation, Flatten, Dropout, Reshape, Lambda
from keras.activations import relu, softmax
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

img_rows, img_cols, ch = 64, 64, 3

def load_image(imagepath):
    imagepath = imagepath.replace(' ', '')
    #image = np.array(Image.open(imagepath))
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def resize_image(image):
    shape = image.shape
    # Crop top and remove hood
    image = image[math.floor(shape[0]/5):shape[0] - 25:,:]
    # Resize to 64 x 64
    image = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    return image

def augment_brightness(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = 0.3 + np.random.uniform()
    new_image[:,:,2] = new_image[:,:,2] * random_brightness
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
    return new_image
    
def augment_trans_shifts(image, steer, trans_range):
    rows, cols, ch = image.shape
    x_translation = trans_range * np.random.uniform() - trans_range / 2
    steer_angle = steer + x_translation / trans_range * 2 * 0.2
    y_translation = 40 * np.random.uniform() - 40 / 2
    
    m_translation = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    image_trans = cv2.warpAffine(image, m_translation, (cols, rows))
    return image_trans, steer_angle

def preprocess(input_files):
    position = np.random.choice(['center', 'left', 'right'])#, p = [0.40, 0.10, 0.50])
    idx = np.random.randint(len(input_files))
    image_path = input_file[position][idx]
    if position == 'left':
        shift_angle = 0.25
    elif position == 'right':
        shift_angle = -0.25
    else:
        shift_angle = 0.

    image = load_image(image_path)
    steer_angle = input_file['steering'][idx] + shift_angle        
    
    image = augment_brightness(image)
    image, steer_angle = augment_trans_shifts(image, steer_angle, 70)
    image = resize_image(image)
    
    if np.random.randint(2) == 0:
        image = cv2.flip(image, 1)
        steer_angle = -steer_angle
    
    return image, steer_angle

def generate_data(input_file, batch_size=32):
    features = np.zeros((batch_size, img_rows, img_cols, ch))
    label = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            X, y = preprocess(input_file)
            X = X.reshape(1, img_rows, img_cols, 3)
            features[i] = X
            label[i] = y
        yield features, label

def get_model():
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', input_shape=(64, 64, 3)))

    model.add(Lambda(lambda x: x / 255. - 0.5))

    model.add(Convolution2D(24, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))
    model.add(Dense(1))

    #model.summary()
    return model

if __name__ == "__main__":
    model = get_model()
    samples_per_epoch = 20000
    nb_epoch = 5
    val_size = samples_per_epoch / 10.

    input_file = pd.read_csv('driving_log.csv')
    data = np.array([resize_image(load_image(img)) for img in input_file['center']])
    y = np.array([angle for angle in input_file['steering']])

    X_test, X_val, y_test, y_val = train_test_split(data, y, test_size=0.33, random_state=42)

    train_generator = generate_data(input_file)
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    model.fit_generator(train_generator,
			samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
			validation_data=(X_val, y_val),
			nb_val_samples=val_size,
			verbose=1)

    score = model.evaluate(X_test, y_test)
    print("Test data %s: %.3f" % (model.metrics_names[1], score[1]))

    # Save model

    json_string = model.to_json()
    model.save_weights('./model.h5')

    import json
    with open("model.json", "w") as json_file:
	    json_file.write(json_string)
