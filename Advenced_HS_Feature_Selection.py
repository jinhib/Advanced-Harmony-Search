import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from keras import backend as K
import time


class K_Fold:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_of_features = len(self.dataset[0, :-1])

    # create ANN model
    def create_model(self):
        # create model
        model = Sequential()
        model.add(Dense(5, input_dim=self.num_of_features, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    # K-fold test
    def k_fold_test(self, k):
        data = self.dataset[:, :-1]
        label = self.dataset[:, -1]

        model = KerasClassifier(build_fn=self.create_model, epochs=200, verbose=0)
        kfold = StratifiedKFold(n_splits=k, shuffle=True)
        # kfold = KFold(n_splits=k, shuffle=True)

        results = cross_val_score(model, data, label, cv=kfold)

        K.clear_session()
        return sum(results)/k


def make_dataset(dataset, h_vector):

    label = dataset[:, -1]
    new_dataset = []

    for idx in enumerate(h_vector):
        if idx[1] == 1:
            if len(new_dataset) == 0:
                new_dataset = dataset[:, idx[0]]
                new_dataset = np.expand_dims(new_dataset, axis=1)
            else:
                data = np.expand_dims(dataset[:, idx[0]], axis=1)
                new_dataset = np.concatenate((new_dataset, data), axis=1)

    label = np.expand_dims(label, axis=1)
    new_dataset = np.concatenate((new_dataset, label), axis=1)

    return new_dataset


# load dataset
# dataset = np.loadtxt('colon_cancer_data.csv', delimiter=",", skiprows=1)
dataset = np.loadtxt('0.82_feature.csv', delimiter=",", skiprows=1, encoding='utf-8')

# number of n(features)
n = len(dataset[0, :-1])
# harmony memory size
hms = 100

# number of iteration
iteration = 500

# Harmony Memory Considering Rate
hmcr = 0.9
# Pitch Adjusting Rate
par = 0.1

number_of_fold = 5

boundary = int(hms * 0.2)

start_time = time.time()

# initialization of harmony memory
hm = np.array([[random.randint(0, 1) for i in range(n + 1)] for j in range(hms)], dtype=float)

for it in range(iteration):

    f = open("HS_Feature_Selection_Result_File_2.txt", 'a')

    # Calculate fitness value
    for h_vector in enumerate(hm):
        data = make_dataset(dataset, h_vector[1][:-1])

        kfold = K_Fold(data)
        hm[h_vector[0]][-1] = kfold.k_fold_test(number_of_fold)

    # upper section harmony memory considering
    new_h_vector = np.array([])
    for idx in range(n):
        # random choice from features
        x = random.choice(hm[:boundary, idx])

        # new_h_vector.append(x)
        new_h_vector = np.append(new_h_vector, x)

    # element for fitness value
    # new_h_vector.append(0)
    new_h_vector = np.append(new_h_vector, 0)

    data = make_dataset(dataset, new_h_vector[:-1])

    kfold = K_Fold(data)
    new_h_vector[-1] = kfold.k_fold_test(number_of_fold)

    # hm.append(new_h_vector)
    new_h_vector = np.expand_dims(new_h_vector, axis=0)
    hm = np.append(hm, new_h_vector, axis=0)

    # lower section harmony memory considering
    if random.uniform(0, 1) < hmcr:
        new_h_vector = np.array([])
        for idx in range(n):
            # random choice from features
            x = random.choice(hm[boundary:, idx])

            # pitch adjusting
            if random.uniform(0, 1) < par:
                if x == 0:
                    x = 1
                else:
                    x = 0

            # new_h_vector.append(x)
            new_h_vector = np.append(new_h_vector, x)

        # element for fitness value
        # new_h_vector.append(0)
        new_h_vector = np.append(new_h_vector, 0)
    else:
        # make a new harmony vector
        new_h_vector = np.array([random.randint(0, 1) for i in range(n + 1)])

    data = make_dataset(dataset, new_h_vector[:-1])

    kfold = K_Fold(data)
    new_h_vector[-1] = kfold.k_fold_test(number_of_fold)

    # hm.append(new_h_vector)
    new_h_vector = np.expand_dims(new_h_vector, axis=0)
    hm = np.append(hm, new_h_vector, axis=0)



    # sort the harmony memory
    hm = sorted(hm, key=lambda x: x[-1], reverse=True)
    hm.pop(-1)
    hm.pop(-1)

    print("Area 1 - Best Accuracy of iteration : " + str(it + 1) + " : " + str(hm[0][-1]))
    print(hm[0][:-1])
    print("Area 2 - Best Accuracy of iteration : " + str(it + 1) + " : " + str(hm[boundary][-1]))
    print(hm[boundary][:-1])

    f.write("Area 1 - Best Accuracy of iteration : " + str(it + 1) + " : " + str(hm[0][-1]) + "\n")
    # f.write(str(hm[0][:-1]) + "\n")
    for i in hm[0][:-1]:
        f.write(str(int(i)) + " ")
    f.write("\n")

    f.write("Area 2 - Best Accuracy of iteration : " + str(it + 1) + " : " + str(hm[boundary][-1]) + "\n")
    # f.write(str(hm[boundary][:-1]) + "\n")
    for i in hm[boundary][:-1]:
        f.write(str(int(i)) + " ")
    f.write("\n")
    f.write("==========================================================" + "\n")
    f.close()

    hm = np.array(hm)

print('Finish')
print('----------------------------------------------------------------')
# best harmony vector
print("Best Harmony vector")
print(hm[0][:-1])

f = open("HS_Feature_Selection_Result_File_2.txt", 'a')
f.write("Accuracy of Best Harmony : " + str(hm[0][-1]) + "\n")
# f.write(hm[0][:-1] + "\n")
for i in hm[0][:-1]:
    f.write(str(int(i)) + " ")
f.write("\n")
f.close()

end_time = time.time()

total_time = end_time - start_time
minute = total_time // 60
total_time = total_time % 60
hour = minute // 60
minute = minute % 60

print("running time : " + str(hour) + "h " + str(minute) + "m " + str(total_time) + "s")
