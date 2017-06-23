import numpy as np
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold

import time


class ConvolutionalNeuralNet(object):

    def mf(self, x, p, y):
        p2 = np.zeros_like(p)

        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)

        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    def optimise_f2_thresholds(self, y, p, verbose=True, resolution=100):

        x = [0.2]*17
        for i in range(17):
            best_i2 = 0
            best_score = 0

            for i2 in range(resolution):
                i2 /= resolution
                x[i] = i2
                score = self.mf(x, p, y)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)

        return x

    def neural_net_train(self, x_train, y_train):

        nfolds = 3
        num_fold = 0
        sum_score = 0

        yfull_test = []
        yfull_train = []

        kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

        for train_index, test_index in kf:

            start_time_model_fitting = time.time()
            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_valid = x_train[test_index]
            Y_valid = y_train[test_index]

            num_fold += 1

            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))

            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

            model = Sequential()
            model.add(BatchNormalization(input_shape=(32, 32, 3)))
            model.add(Conv2D(8, 1, 1, activation='relu'))
            model.add(Conv2D(16, 2, 2, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, 3, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, 3, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(17, activation='sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=2, nb_epoch=10, callbacks=callbacks,
              shuffle=True)

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=128, verbose=2)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
    print("Optimizing prediction threshold")
    print(optimise_f2_thresholds(Y_valid, p_valid))

    p_test = model.predict(x_train, batch_size=128, verbose=2)
    yfull_train.append(p_test)

    p_test = model.predict(x_test, batch_size=128, verbose=2)
    yfull_test.append(p_test)

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)