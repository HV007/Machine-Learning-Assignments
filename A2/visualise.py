import numpy as np
import pandas as pd
from PIL import Image as im

def load_data(path):
    data = pd.read_csv(path).to_numpy()
    X = data[:,:-1]
    y = data[:,-1]
    y = np.expand_dims(y, axis = 1)
    return X, y

if __name__ == '__main__':
    pred = np.load('pred_libsvm.npy')
    X, y = load_data('mnist/test.csv')
    count = 0
    for i in range(len(y)):
        if y[i] != pred[i]:
            count += 1
            temp = np.reshape(X[i].astype(np.uint8), (28, 28))
            data = im.fromarray(temp)
            data = data.convert('RGB')
            data.save('images/' + str(count) + '.png')
            print(pred[i])
        if count >= 10:
            break

