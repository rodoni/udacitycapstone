import numpy as np
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold

import time
import os


class ConvNet(object):

    def model_arquiteture(self):

        model = Sequential()
        model.add(BatchNormalization(input_shape=(32, 32, 3)))
        model.add(Conv2D(8, (1, 1), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    def neural_net_train(self, x_train, y_train, n_batch):

        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(n_batch) + '.h5')

        model = self.model_arquiteture()

        if n_batch > 0:
            kfold_weights_path_p = os.path.join('', 'weights_kfold_' + str(n_batch - 1) + '.h5')
            if os.path.isfile(kfold_weights_path_p):
                model.load_weights(kfold_weights_path_p)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        model.fit(x=x_train, y=y_train, validation_split=0.1,
                  batch_size=128, verbose=2, epochs=10, callbacks=callbacks,
                  shuffle=True)

        return model

    def neural_net_predict(self, x_test, last_batch):

        output = []
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(last_batch) + '.h5')

        model = self.model_arquiteture()

        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        p_test = model.predict(x_test, batch_size=128, verbose=2)
        output.extend(p_test)

        return output
