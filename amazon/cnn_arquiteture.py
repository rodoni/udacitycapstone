import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
import os


class ConvNet(object):

    @staticmethod
    def model_arquiteture():

        """
        :return: CCN Model
        """

        model = Sequential()
        model.add(BatchNormalization(input_shape=(64, 64, 3)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer='glorot_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='glorot_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer='glorot_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer='glorot_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu',kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    def neural_net_train(self, x_train, y_train, n_folds):

        """
        :param x_train: matrix of images to train
        :param y_train: labels 
        :param n_folds: number of folds 
        :return: 
        """

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
        fold = 0

        x_train = np.array(x_train)
        thresholds_list = []
        fb_score_list = []

        for train_index, test_index in kf.split(x_train):

            fold += 1
            print('Training KFold number {} from {}'.format(fold, n_folds))
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(fold) + '.h5')
            model = self.model_arquiteture()
            history_array = []

            epochs_arr = [20, 5, 5]
            learn_rates = [0.001, 0.0001, 0.00001]

            for learn_rate, epochs in zip(learn_rates, epochs_arr):
                opt = optimizers.Nadam(lr=learn_rate)
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
                callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                             ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
                hst = model.fit(x=x_train[train_index], y=y_train[train_index], validation_data=(x_train[test_index], y_train[test_index]),
                          batch_size=64, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)
                history_array.append(hst)

            p_valid = model.predict(x_train[test_index], batch_size=64)
            thresholds = self.optimise_f2_thresholds(np.array(y_train[test_index]), np.array(p_valid), verbose=False,
                                                     resolution=100)
            thresholds_list.append(thresholds)
            fb_score = fbeta_score(np.array(y_train[test_index]), np.array(p_valid) > thresholds, beta=2, average='samples')
            fb_score_list.append(fb_score)

            print('Fbeta Score  KFold number {} from {} : {}'.format(fold, n_folds, fb_score))
            print('Threshold: {}'.format(thresholds))

        fb_score_max = np.array(fb_score_list[0])
        threshold = np.array(thresholds_list[0])
        for i in range(1, n_folds):
            if fb_score_max < np.array(fb_score_list[i]):
                threshold = np.array(thresholds_list[i])

        return threshold

    def neural_net_predict_fold(self, x_test, n_folds):

        """
        :param x_test: matrix of images to predict
        :param n_folds: number of folds , must be the same used in neural_net_train
        :return: matrix with mean of all folds 
        """

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

    @staticmethod
    def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
        def mf(x):
            p2 = np.zeros_like(p)
            for i in range(17):
                p2[:, i] = (p[:, i] > x[i]).astype(np.float)
            score = fbeta_score(y, p2, beta=2, average='samples')
            return score

        x = [0.23] * 17
        for i in range(17):
            best_i2 = 0
            best_score = 0
            for i2 in range(resolution):
                i2 /= float(resolution)
                x[i] = i2
                score = mf(x)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)

        return x

