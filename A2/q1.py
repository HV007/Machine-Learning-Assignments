import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import math
import sys

SUMMARY_MULTIPLIER = 5

word_to_id = {}
ps = PorterStemmer()
wl = WordNetLemmatizer()

def load_data(path, raw = True, lem = False, trigram = False, sum = False):
    f = open(path)
    lines = f.readlines()
    # lines = lines[:100]
    reviews = []
    ratings = []
    summary = []
    for line in lines:
        data = json.loads(line)
        reviews.append(data['reviewText'])
        ratings.append(int(data['overall']))
        summary.append(data['summary'])
    f.close()
    stpwords = set(stopwords.words('english'))
    processed_reviews = []
    for idx, review in enumerate(reviews):
        words = review.split()
        temp = []
        for word in words:
            if not raw:
                if word not in stpwords and word != '':
                    word = ps.stem(word)
                    if lem:
                        word = wl.lemmatize(word)
                    temp.append(word)
            else :
                temp.append(word)
        if sum:
            for word in summary[idx]:
                if not raw:
                    if word not in stpwords and word != '':
                        word = ps.stem(word)
                        if lem:
                            word = wl.lemmatize(word)
                        for _ in range(SUMMARY_MULTIPLIER):
                            temp.append(word)
                else :
                    for _ in range(SUMMARY_MULTIPLIER):
                        temp.append(word)
        if trigram:
            temp1 = []
            for i in range(2, len(temp)):
                temp1.append(temp[i-2] + ' ' + temp[i-1] + ' ' + temp[i])
            temp = temp1
        processed_reviews.append(temp)

    return processed_reviews, ratings

def encode_reviews(reviews, ratings):
    id = 0
    count = 0
    for words in reviews:
        for word in words:
            if word not in word_to_id:
                word_to_id[word] = id
                id += 1
    X = []
    for words in reviews:
        temp = []
        for word in words:
            temp.append(word_to_id[word])
        X.append(temp)
    y = np.array(ratings)
    y -= 1
    return X, y, id

def get_phi_y(i, y):
    m = y.shape[0]
    return np.count_nonzero(y == i)/m

def get_paramaters(X, y, V):
    phi_x = [[1 for j in range(5)] for i in range(V)]
    phi_y = [get_phi_y(i, y) for i in range(5)]
    temp = [V for i in range(5)]
    m = len(X)
    for k in range(m):
        temp[y[k]] += len(X[k])
        for l in range(len(X[k])):
            phi_x[X[k][l]][y[k]] += 1
    for k in range(5):
        for l in range(V):
            phi_x[l][k] /= temp[k]
    return phi_x, phi_y

def predict(X, phi_x, phi_y):
    m = len(X)
    y_pred = []
    for i in range(m):
        max_prob = -float('inf')
        pred = -1
        for j in range(5):
            prob = math.log(phi_y[j])
            for k in range(len(X[i])):
                prob += math.log(phi_x[X[i][k]][j])
            if prob > max_prob:
                max_prob = prob
                pred = j
        y_pred.append(pred)
    return np.array(y_pred)

def get_accuracy(y, y_pred):
    m = len(y)
    return np.count_nonzero(y == y_pred)/m

def get_test_accuracy(phi_x, phi_y, reviews, ratings):
    X = []
    for words in reviews:
        temp = []
        for word in words:
            if word not in word_to_id:
                continue
            temp.append(word_to_id[word])
        X.append(temp)
    y = np.array(ratings)
    y -= 1
    y_pred = predict(X, phi_x, phi_y)
    m = len(y)
    return X, y, y_pred, np.count_nonzero(y == y_pred)/m

def get_random_accuracy(y):
    m = len(y)
    y_pred = np.random.randint(0, 5, (m,))
    return np.count_nonzero(y == y_pred)/m

def get_max_accuracy(y, y_test):
    y_pred = np.bincount(y).argmax()
    return np.count_nonzero(np.expand_dims(y_test, axis = 1) == y_pred)/len(y_test)

def get_confusion_matrix(y, y_pred):
    C = np.zeros((5, 5))
    m = len(y)
    for i in range(5):
        for j in range(5):
            for k in range(m):
                if y_pred[k] == j and y[k] == i:
                    C[i][j] += 1
    return C

def get_f1_score(C):
    scores = []
    for i in range(5):
        tp = C[i][i]
        fp = 0
        fn = 0
        for j in range(5):
            if i == j:
                continue
            fp += C[i][j]
            fn += C[j][i]
        scores.append(2*tp / (2*tp + fp + fn))
    return np.array(scores)

if __name__ == '__main__':
    TRAIN_PATH = sys.argv[1]
    TEST_PATH = sys.argv[2]

    if sys.argv[3] == 'a':
        reviews, ratings = load_data(TRAIN_PATH)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        y_pred = predict(X, phi_x, phi_y)
        acc = get_accuracy(y, y_pred)
        print('Training Accuracy: ', acc)

        reviews, ratings = load_data(TEST_PATH)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        print('Test Accuraacy: ', acc)

        C = get_confusion_matrix(y_test, y_pred_test)
        f1_scores = get_f1_score(C)
        print('F1 Score: ', f1_scores)
        print('Macro F1 Score: ', np.mean(f1_scores))

    elif sys.argv[3] == 'b':
        reviews, ratings = load_data(TRAIN_PATH)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        acc = get_random_accuracy(y_test)
        print('Random Accuracy: ', acc)

        acc = get_max_accuracy(y, y_test)
        print('Max Accuracy: ', acc)

    elif sys.argv[3] == 'c':
        reviews, ratings = load_data(TRAIN_PATH)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        reviews, ratings = load_data(TEST_PATH)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)

        C = get_confusion_matrix(y_test, predict(X_test, phi_x, phi_y))
        np.set_printoptions(suppress = True)
        print('Confusion Matrix: ', C)
        
    elif sys.argv[3] == 'd':
        reviews, ratings = load_data(TRAIN_PATH, raw = False)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        y_pred = predict(X, phi_x, phi_y)
        acc = get_accuracy(y, y_pred)
        print('Training Accuracy: ', acc)

        reviews, ratings = load_data(TEST_PATH, raw = False)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        print('Test Accuraacy: ', acc)

        C = get_confusion_matrix(y_test, y_pred_test)
        f1_scores = get_f1_score(C)
        print('F1 Score: ', f1_scores)
        print('Macro F1 Score: ', np.mean(f1_scores))

    elif sys.argv[3] == 'e':
        reviews, ratings = load_data(TRAIN_PATH, trigram = True)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        y_pred = predict(X, phi_x, phi_y)
        acc = get_accuracy(y, y_pred)
        print('Training Accuracy with Tigram: ', acc)

        reviews, ratings = load_data(TEST_PATH, trigram = True)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        print('Test Accuraacy with Tigram: ', acc)

        C = get_confusion_matrix(y_test, y_pred_test)
        f1_scores = get_f1_score(C)
        print('F1 Score: ', f1_scores)
        print('Macro F1 Score: ', np.mean(f1_scores))

        reviews, ratings = load_data(TRAIN_PATH, raw = False, lem = True)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        y_pred = predict(X, phi_x, phi_y)
        acc = get_accuracy(y, y_pred)
        print('Training Accuracy with Lemmatization: ', acc)

        reviews, ratings = load_data(TEST_PATH, raw = False, lem = True)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        print('Test Accuracy with Lemmatization: ', acc)

        C = get_confusion_matrix(y_test, y_pred_test)
        f1_scores = get_f1_score(C)
        print('F1 Score: ', f1_scores)
        print('Macro F1 Score: ', np.mean(f1_scores))

    elif sys.argv[3] == 'g':
        reviews, ratings = load_data(TRAIN_PATH, sum = True)
        X, y, V = encode_reviews(reviews, ratings)
        phi_x, phi_y = get_paramaters(X, y, V)

        y_pred = predict(X, phi_x, phi_y)
        acc = get_accuracy(y, y_pred)
        print('Training Accuracy: ', acc)

        reviews, ratings = load_data(TEST_PATH, sum = True)
        X_test, y_test, y_pred_test, acc = get_test_accuracy(phi_x, phi_y, reviews, ratings)
        print('Test Accuracy: ', acc)

        C = get_confusion_matrix(y_test, y_pred_test)
        f1_scores = get_f1_score(C)
        print('F1 Score: ', f1_scores)
        print('Macro F1 Score: ', np.mean(f1_scores))