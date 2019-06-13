import csv
import matplotlib.image as mpimg
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
 
lines=[]
with open("./data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)



lines=lines[1:]
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("Training Data Length: ",len(train_samples))
print("Validation Data Length: ",len(validation_samples))



#data generation and preprocessing

import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam

def generator(samples, batch_size=32,correctionFactor=0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                #center
                name ='./data/IMG/'+batch_sample[0].split('/')[-1]
                center_image =mpimg.imread(name)
                #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                centralAngle=float(batch_sample[3])

                #left
                name = './data/IMG/' + batch_sample[1].split('/')[-1]
                left_image = mpimg.imread(name)
                #left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + correctionFactor

                #right
                name = './data/IMG/' + batch_sample[2].split('/')[-1]
                right_image = mpimg.imread(name)
                #right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - correctionFactor


                images.append(center_image)
                angles.append(centralAngle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

                # Augment Data by flipping
                flipped_images, flipped_angles = [], []
                for image, measurement in zip(images, angles):
                    flipped_images.append(image)
                    flipped_angles.append(measurement)
                    flipped_images.append(np.fliplr(image))
                    flipped_angles.append(measurement * -1.0)

                X_train = np.array(flipped_images)
                y_train = np.array(flipped_angles)

                yield shuffle(X_train, y_train)

#generator call for training and validation set
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#model definition:using comma ai architecture

"""model = Sequential()

model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))"""

#nvidia

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.50))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.20))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
model.compile(optimizer = "adam", loss = "mse")
history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('nvidiamodel1.h5')




