import numpy as np
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import KFold


import os


class ConvNet(object):

    def model_arquiteture(self):

        model = Sequential()
        model.add(BatchNormalization(input_shape=(64, 64, 3)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    def neural_net_train(self, x_train, y_train, n_folds):

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
        fold = 0

        x_train = np.array(x_train)
        print(x_train.shape)

        for train_index, test_index in kf.split(x_train):

            fold += 1
            print('Training KFold number {} from {}'.format(fold, n_folds))
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(fold) + '.h5')
            model = self.model_arquiteture()

            epochs_arr = [20, 5, 5]
            learn_rates = [0.001, 0.0001, 0.00001]

            for learn_rate, epochs in zip(learn_rates, epochs_arr):
                opt = optimizers.Nadam(lr=learn_rate)
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
                callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                             ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
                model.fit(x=x_train[train_index], y=y_train[train_index], validation_data=(x_train[test_index], y_train[test_index]),
                          batch_size=128, verbose=2, epochs=epochs, callbacks=callbacks,
                          shuffle=True)

        return model

    def neural_net_predict_fold(self, x_test, n_folds):

        num_fold = 0
        all_test = []

        for i in range(n_folds):
            num_fold += 1
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(i) + '.h5')
            model = self.model_arquiteture()
            print('Start KFold number {} from {}'.format(num_fold, n_folds))
            if os.path.isfile(kfold_weights_path):
                model.load_weights(kfold_weights_path)
            test = model.predict(x_test.astype('float32'), batch_size=128, verbose=2)
            all_test.append(test)

        # merge all folds
        mean = np.array(all_test[0])
        for i in range(1, n_folds):
            mean += np.array(all_test[i])
        mean /= n_folds

        return mean

    def neural_net_predict(self, x_test, last_batch):

        output = []
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(last_batch) + '.h5')

        model = self.model_arquiteture()

        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        p_test = model.predict(x_test, batch_size=128, verbose=2)
        output.extend(p_test)

        return output
