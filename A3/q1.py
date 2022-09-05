import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class Node:
    def __init__(self, data, data_valid, data_test):
        self.data = data
        self.data_valid = data_valid
        self.data_test = data_test
        self.children = []
        self.split_index = None
        self.split_values = []
        self.leaf = True
        self.prediction = None
        self.correct = 0
        self.correct_valid = 0
        self.correct_test = 0

NUM_ATTRIBUTES = 17
THRESHOLD = 10
is_numeric = [True, False, False, False, False, True, False, False, False, True, False, True, True, True, True, False, False]

word_to_id = [{} for i in range(NUM_ATTRIBUTES)]
id_to_word = [{} for i in range(NUM_ATTRIBUTES)]
id = [0 for i in range(NUM_ATTRIBUTES)]

best_node = None
best_improvement = 0

def load_and_encode_data(path):
    f = open(path)
    lines = f.readlines()
    lines = lines[1:]
    data = []
    for line in lines:
        words = line.split(';')
        temp = []
        for j in range(NUM_ATTRIBUTES):
            word = words[j].strip('\"').strip('\n')
            if is_numeric[j]:
                temp.append(float(word))
            elif word not in word_to_id[j]:
                word_to_id[j][word] = id[j]
                id_to_word[j][id[j]] = word
                temp.append(id[j])
                id[j] += 1
            else :
                temp.append(word_to_id[j][word])
        data.append(temp)
    return np.array(data)

def one_hot_encode_data(data):
    data_new = []
    for i in range(len(data)):
        temp = []
        for j in range(NUM_ATTRIBUTES):
            if is_numeric[j] or id[j] <= 2:
                temp.append(data[i][j])
            else :
                for k in range(id[j]):
                    if data[i][j] == k:
                        temp.append(1)
                    else :
                        temp.append(0)
        data_new.append(temp)
    return np.array(data_new)

def update_globals():
    global NUM_ATTRIBUTES, id, is_numeric
    is_numeric_new = []
    id_new = []
    num_attr = 49
    for j in range(NUM_ATTRIBUTES):
        if is_numeric[j]:
            is_numeric_new.append(True)
        elif id[j] <= 2:
            is_numeric_new.append(False)
        else :
            for k in range(id[j]):
                is_numeric_new.append(False)
    for j in range(NUM_ATTRIBUTES):
        if is_numeric[j]:
            id_new.append(0)
        elif id[j] <= 2:
            id_new.append(id[j])
        else :
            for k in range(id[j]):
                id_new.append(2)
    
    NUM_ATTRIBUTES = num_attr
    id = id_new
    is_numeric = is_numeric_new

def get_median(data, idx):
    return np.median(data[:,idx])

def get_h(data):
    if len(data) == 0:
        return 0
    prob_ones = np.count_nonzero(data[:,-1] == 1) / len(data)
    prob_zeros = np.count_nonzero(data[:,-1] == 0) / len(data)
    if prob_ones == 0 or prob_zeros == 0:
        return 0
    return -prob_ones * np.log(prob_ones) - prob_zeros * np.log(prob_zeros)

def get_mutual_information(data, idx):
    h_y_given_x = 0
    if is_numeric[idx]:
        med = get_median(data, idx)
        h_y_given_x += (np.count_nonzero(data[:,idx] <= med) / len(data)) * get_h(data[data[:,idx] <= med])
        h_y_given_x += (np.count_nonzero(data[:,idx] > med) / len(data)) * get_h(data[data[:,idx] > med])
    else :
        for k in range(id[idx]):
            prob = np.count_nonzero(data[:,idx] == k) / len(data)
            h_y_given_x += prob * get_h(data[data[:,idx] == k])
    return h_y_given_x

def choose_attribute(data):
    min_entropy = float('inf')
    attr = -1
    for idx in range(NUM_ATTRIBUTES - 1):
        if get_mutual_information(data, idx) < min_entropy:
            min_entropy = get_mutual_information(data, idx)
            attr = idx
    return attr

def get_prediction(data):
    return 1 if np.count_nonzero(data[:,-1] == 1) > np.count_nonzero(data[:,-1] == 0) else 0

def get_correct(data, pred):
    return np.count_nonzero(data[:,-1] == pred)

def get_data_split(node, idx, l, r):
    data_split = node.data[np.all(np.column_stack((node.data[:,idx] > l, node.data[:,idx] <= r)), axis = 1)]
    data_valid_split = node.data_valid[np.all(np.column_stack((node.data_valid[:,idx] > l, node.data_valid[:,idx] <= r)), axis = 1)]
    data_test_split = node.data_test[np.all(np.column_stack((node.data_test[:,idx] > l, node.data_test[:,idx] <= r)), axis = 1)]
    return data_split, data_valid_split, data_test_split

def generate_tree(data, data_valid, data_test, max_nodes):
    root = Node(data, data_valid, data_test)
    root.prediction = get_prediction(data)
    root.correct = get_correct(data, root.prediction)
    root.correct_valid = get_correct(data_valid, root.prediction)
    root.correct_test = get_correct(data_test, root.prediction)
    queue = []
    queue.append(root)
    num_nodes = 1

    num_nodes_list = [1]
    train_correct_list = [root.correct]
    valid_correct_list = [root.correct_valid]
    test_correct_list = [root.correct_test]

    while len(queue) > 0 and max_nodes > num_nodes:
        node = queue.pop(0)
        if node.correct == len(node.data):
            continue
        if len(node.data) < THRESHOLD:
            continue
        node.leaf = False
        split_index = choose_attribute(node.data)
        correct_improvement = 0
        correct_valid_improvement = 0
        correct_test_improvement = 0
        if is_numeric[split_index]:
            med = get_median(node.data, split_index)
            node.split_values.append(med)

            data_left, data_left_valid, data_left_test = get_data_split(node, split_index, -float('inf'), med)
            if len(data_left) == len(data) or len(data_left) == 0:
                node.leaf = True
                continue
            node_left = Node(data_left, data_left_valid, data_left_test)
            node_left.prediction = get_prediction(data_left)
            node_left.correct = get_correct(data_left, node_left.prediction)
            node_left.correct_valid = get_correct(data_left_valid, node_left.prediction)
            node_left.correct_test = get_correct(data_left_test, node_left.prediction)
            correct_improvement += node_left.correct
            correct_valid_improvement += node_left.correct_valid
            correct_test_improvement += node_left.correct_test
            node.children.append(node_left)
            queue.append(node_left)
            num_nodes += 1

            data_right, data_right_valid, data_right_test = get_data_split(node, split_index, med, float('inf'))
            node_right = Node(data_right, data_right_valid, data_right_test)
            node_right.prediction = get_prediction(data_right)
            node_right.correct = get_correct(data_right, node_right.prediction)
            node_right.correct_valid = get_correct(data_right_valid, node_right.prediction)
            node_right.correct_test = get_correct(data_right_test, node_right.prediction)
            correct_improvement += node_right.correct
            correct_valid_improvement += node_right.correct_valid
            correct_test_improvement += node_right.correct_test
            node.children.append(node_right)
            queue.append(node_right)
            num_nodes += 1
        else :
            for k in range(id[split_index]):
                node.split_values.append(k)
                data_new, data_new_valid, data_new_test = get_data_split(node, split_index, k - 0.5, k + 0.5)
                node_new = Node(data_new, data_new_valid, data_new_test)
                node_new.prediction = get_prediction(data_new)
                node_new.correct = get_correct(data_new, node_new.prediction)
                node_new.correct_valid = get_correct(data_new_valid, node_new.prediction)
                node_new.correct_test = get_correct(data_new_test, node_new.prediction)
                correct_improvement += node_new.correct
                correct_valid_improvement += node_new.correct_valid
                correct_test_improvement += node_new.correct_test
                node.children.append(node_new)
                queue.append(node_new)
                num_nodes += 1
        correct_improvement -= node.correct
        correct_valid_improvement -= node.correct_valid
        correct_test_improvement -= node.correct_test

        num_nodes_list.append(num_nodes)
        train_correct_list.append(train_correct_list[-1] + correct_improvement)
        valid_correct_list.append(valid_correct_list[-1] + correct_valid_improvement)
        test_correct_list.append(test_correct_list[-1] + correct_test_improvement)
    
    train_correct_list = [train_elem / len(data) for train_elem in train_correct_list]
    valid_correct_list = [valid_elem / len(data_valid) for valid_elem in valid_correct_list]
    test_correct_list = [test_elem / len(data_test) for test_elem in test_correct_list]
    
    return num_nodes_list, train_correct_list, valid_correct_list, test_correct_list, root

def prune_tree(root, num_nodes_head, train_correct_head, valid_correct_head, test_correct_head):
    global best_node, best_improvement
    num_nodes_list = [num_nodes_head]
    train_correct_list = [train_correct_head * len(root.data)]
    valid_correct_list = [valid_correct_head * len(root.data_valid)]
    test_correct_list = [test_correct_head * len(root.data_test)]

    def recurse(root):
        total_children_train = root.correct
        total_children_valid = root.correct_valid
        total_children_test = root.correct_test
        count = 1
        for child in root.children:
            child_correct, child_correct_valid, child_correct_test, child_count = recurse(child)
            total_children_train += child_correct
            total_children_valid += child_correct_valid
            total_children_test += child_correct_test
            count += child_count
        return total_children_train, total_children_valid, total_children_test, count

    def get_improvement(root):
        total_children_train = 0
        total_children_valid = 0
        total_children_test = 0
        count = 1
        if root.leaf:
            return 0, 0, 0, 1
        for child in root.children:
            child_correct, child_correct_valid, child_correct_test, child_count = recurse(child)
            total_children_train += child_correct
            total_children_valid += child_correct_valid
            total_children_test += child_correct_test
            count += child_count
        return root.correct - total_children_train, root.correct_valid - total_children_valid, root.correct_test - total_children_test, count

    def get_best_improvement(root):
        global best_node, best_improvement
        train_improvement, valid_improvement, test_improvement, count = get_improvement(root)
        if valid_improvement >= best_improvement:
            best_improvement = valid_improvement
            best_node = root
        for child in root.children:
            get_best_improvement(child)
    
    while True:
        best_node = None
        best_improvement = 0
        get_best_improvement(root)
        if best_node == None or best_node.leaf:
            break
        train_improvement, valid_improvement, test_improvement, count = get_improvement(best_node)
        num_nodes_list.append(num_nodes_list[-1] - count + 1)
        train_correct_list.append(train_correct_list[-1] + train_improvement)
        valid_correct_list.append(valid_correct_list[-1] + valid_improvement)
        test_correct_list.append(test_correct_list[-1] + test_improvement)
        best_node.children = []
        best_node.split_values = []
        best_node.leaf = True

    # def recurse(root):
    #     total_children_train = 0
    #     total_children_valid = 0
    #     total_children_test = 0
    #     count = 1
    #     if root.leaf:
    #         train_correct = root.correct
    #         valid_correct = root.correct_valid
    #         test_correct = root.correct_test
    #         return train_correct, valid_correct, test_correct, 1
    #     for child in root.children:
    #         recurse(child)
    #         child_correct, child_correct_valid, child_correct_test, child_count = recurse(child)
    #         total_children_train += child_correct
    #         total_children_valid += child_correct_valid
    #         total_children_test += child_correct_test
    #         count += child_count
    #     if root.correct_valid >= total_children_valid:
    #         num_nodes_list.append(num_nodes_list[-1] - count + 1)
    #         train_correct_list.append(train_correct_list[-1] + root.correct - total_children_train)
    #         valid_correct_list.append(valid_correct_list[-1] + root.correct_valid - total_children_valid)
    #         test_correct_list.append(test_correct_list[-1] + root.correct_test - total_children_test)
    #         root.children = []
    #         root.split_values = []
    #         root.leaf = True
    #         return root.correct, root.correct_valid, root.correct_test, 1
    #     return total_children_train, total_children_valid, total_children_test, count
        
    # recurse(root)

    train_correct_list = [train_elem / len(root.data) for train_elem in train_correct_list]
    valid_correct_list = [valid_elem / len(root.data_valid) for valid_elem in valid_correct_list]
    test_correct_list = [test_elem / len(root.data_test) for test_elem in test_correct_list]
    return num_nodes_list, train_correct_list, valid_correct_list, test_correct_list

def plot_accuracy(num_nodes_list, train_correct_list, valid_correct_list, test_correct_list):
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy')
    plt.plot(num_nodes_list, train_correct_list, label = 'Training Accruacy')
    plt.plot(num_nodes_list, valid_correct_list, label = 'Validation Accruacy')
    plt.plot(num_nodes_list, test_correct_list, label = 'Test Accruacy')
    plt.legend()

def partA(train_path, valid_path, test_path, one_hot = False):
    data = load_and_encode_data(train_path)
    data_valid = load_and_encode_data(valid_path)
    data_test = load_and_encode_data(test_path)

    if one_hot:
        data = one_hot_encode_data(data)
        data_valid = one_hot_encode_data(data_valid)
        data_test = one_hot_encode_data(data_test)
        update_globals()

    num_nodes_list, train_correct_list, valid_correct_list, test_correct_list, root = generate_tree(data, data_valid, data_test, 10000)
    print(train_correct_list[-1], valid_correct_list[-1], test_correct_list[-1])
    plot_accuracy(num_nodes_list, train_correct_list, valid_correct_list, test_correct_list)
    plt.show()

def partB(train_path, valid_path, test_path, one_hot = False):
    data = load_and_encode_data(train_path)
    data_valid = load_and_encode_data(valid_path)
    data_test = load_and_encode_data(test_path)

    if one_hot:
        data = one_hot_encode_data(data)
        data_valid = one_hot_encode_data(data_valid)
        data_test = one_hot_encode_data(data_test)
        update_globals()

    num_nodes_list, train_correct_list, valid_correct_list, test_correct_list, root = generate_tree(data, data_valid, data_test, 10000)
    plot_accuracy(num_nodes_list, train_correct_list, valid_correct_list, test_correct_list)
    num_nodes_list, train_correct_list, valid_correct_list, test_correct_list = prune_tree(root, num_nodes_list[-1], train_correct_list[-1], valid_correct_list[-1], test_correct_list[-1])
    print(train_correct_list[-1], valid_correct_list[-1], test_correct_list[-1])
    print(num_nodes_list[-1])
    plot_accuracy(num_nodes_list, train_correct_list, valid_correct_list, test_correct_list)
    plt.show()

def partC(train_path, valid_path, test_path):
    data = load_and_encode_data(train_path)
    data_valid = load_and_encode_data(valid_path)
    data_test = load_and_encode_data(test_path)

    data = one_hot_encode_data(data)
    data_valid = one_hot_encode_data(data_valid)
    data_test = one_hot_encode_data(data_test)
    update_globals()

    best_oob = 0
    best_model = None
    best_n_estimators = -1
    best_max_features = -1
    best_min_samples_split = -1

    for n_estimators in range(50, 451, 100):
        for max_features in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for min_samples_split in range(2, 11, 2):
                print(n_estimators, max_features, min_samples_split)
                clf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features, min_samples_split = min_samples_split, oob_score = True)
                clf.fit(data[:,:-1], data[:,-1])
                if clf.oob_score_ > best_oob:
                    best_oob = clf.oob_score_
                    best_model = clf
                    best_n_estimators = n_estimators
                    best_max_features = max_features
                    best_min_samples_split = min_samples_split
    
    best_train = np.count_nonzero(data[:,-1] == best_model.predict(data[:,:-1])) / len(data)
    best_valid = np.count_nonzero(data_valid[:,-1] == best_model.predict(data_valid[:,:-1])) / len(data_valid)
    best_test = np.count_nonzero(data_test[:,-1] == best_model.predict(data_test[:,:-1])) / len(data_test)
    
    print(best_train, best_valid, best_test, best_oob)
    print(best_n_estimators, best_max_features, best_min_samples_split)

def partD(train_path, valid_path, test_path):
    data = load_and_encode_data(train_path)
    data_valid = load_and_encode_data(valid_path)
    data_test = load_and_encode_data(test_path)

    data = one_hot_encode_data(data)
    data_valid = one_hot_encode_data(data_valid)
    data_test = one_hot_encode_data(data_test)
    update_globals()

    best_n_estimators = 450
    best_max_features = 0.7
    best_min_samples_split = 10

    fig = plt.figure()
    param_values = []
    valid_accs = []
    test_accs = []
    for n_estimators in range(50, 451, 100):
        print(n_estimators)
        clf = RandomForestClassifier(n_estimators = n_estimators, max_features = best_max_features, min_samples_split = best_min_samples_split)
        clf.fit(data[:,:-1], data[:,-1])
        valid_acc = np.count_nonzero(data_valid[:,-1] == clf.predict(data_valid[:,:-1])) / len(data_valid)
        test_acc = np.count_nonzero(data_test[:,-1] == clf.predict(data_test[:,:-1])) / len(data_test)
        param_values.append(n_estimators)
        valid_accs.append(valid_acc)
        test_accs.append(test_acc)
    plt.plot(param_values, valid_accs, label = 'Validation Accuracy')
    plt.plot(param_values, test_accs, label = 'Test Accuracy')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    fig = plt.figure()
    param_values = []
    valid_accs = []
    test_accs = []
    for max_features in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(max_features)
        clf = RandomForestClassifier(n_estimators = best_n_estimators, max_features = max_features, min_samples_split = best_min_samples_split)
        clf.fit(data[:,:-1], data[:,-1])
        valid_acc = np.count_nonzero(data_valid[:,-1] == clf.predict(data_valid[:,:-1])) / len(data_valid)
        test_acc = np.count_nonzero(data_test[:,-1] == clf.predict(data_test[:,:-1])) / len(data_test)
        param_values.append(max_features)
        valid_accs.append(valid_acc)
        test_accs.append(test_acc)
    plt.plot(param_values, valid_accs, label = 'Validation Accuracy')
    plt.plot(param_values, test_accs, label = 'Test Accuracy')
    plt.xlabel('Maximum Features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    fig = plt.figure()
    param_values = []
    valid_accs = []
    test_accs = []
    for min_samples_split in range(2, 11, 2):
        print(min_samples_split)
        clf = RandomForestClassifier(n_estimators = best_n_estimators, max_features = best_max_features, min_samples_split = min_samples_split)
        clf.fit(data[:,:-1], data[:,-1])
        valid_acc = np.count_nonzero(data_valid[:,-1] == clf.predict(data_valid[:,:-1])) / len(data_valid)
        test_acc = np.count_nonzero(data_test[:,-1] == clf.predict(data_test[:,:-1])) / len(data_test)
        param_values.append(min_samples_split)
        valid_accs.append(valid_acc)
        test_accs.append(test_acc)
    plt.plot(param_values, valid_accs, label = 'Validation Accuracy')
    plt.plot(param_values, test_accs, label = 'Test Accuracy')
    plt.xlabel('Minimum Samples Split')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train_path, valid_path, test_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if sys.argv[4] == 'a':
        partA(train_path, valid_path, test_path)
    elif sys.argv[4] == 'b':
        partB(train_path, valid_path, test_path)
    elif sys.argv[4] == 'c':
        partC(train_path, valid_path, test_path)
    elif sys.argv[4] == 'd':
        partD(train_path, valid_path, test_path)