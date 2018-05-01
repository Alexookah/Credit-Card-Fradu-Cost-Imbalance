import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE


dict_classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Linear SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

number_of_classifiers = len(dict_classifiers.keys())


def load_dataset():
    # Import and normalize the data
    data = pd.read_csv('creditcard.csv')
    # drop un-needed Time column
    data.drop(["Time"], axis=1, inplace=True)

    count_normal_transaction = len(data[data["Class"] == 0])  # normal transaction are repersented by 0
    count_fraud_transaction = len(data[data["Class"] == 1])  # fraud by 1
    percentage_of_normal_transaction = count_normal_transaction / (
            count_normal_transaction + count_fraud_transaction)
    percentage_of_fraud_transaction = count_fraud_transaction / (count_normal_transaction + count_fraud_transaction)

    print("percentage of normal transaction is", percentage_of_normal_transaction * 100)
    print("percentage of fraud transaction", percentage_of_fraud_transaction * 100)

    return data


def train_data(classifier, label, x_train, y_train, x_test, y_test):
    df_results = pd.DataFrame(data=np.zeros(shape=(0, 4)),
                              columns=['classifier', 'train_score', 'test_score', 'training_time'])
    print("Running... Train data with Classifier: ", label)
    t_start = time.clock()

    classifier.fit(x_train, y_train.values.ravel())
    t_end = time.clock()
    t_diff = t_end - t_start
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)

    df_results.loc[0, 'classifier'] = label
    df_results.loc[0, 'train_score'] = train_score
    df_results.loc[0, 'test_score'] = test_score
    df_results.loc[0, 'training_time'] = t_diff
    print("trained {c} in {f:.2f} s".format(c=label, f=t_diff))



def cross_my_val_score(classifiers, x_cross_val, y_cross_val):
    best_score = 0.0
    best_classifier = ""
    for classifier, label in classifiers:
        print("Running... Cross validation with Classifier: ", label)
        start = time.time()
        scores = cross_val_score(classifier, x_cross_val, y_cross_val, cv=3)
        stop = time.time()
        print("Cross - validated scores", scores)
        print("%20s Accuracy: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_classifier = label

    print("Best Score: ", best_score)
    print("Best Classifier: ", best_classifier)
    return best_classifier


def model(model,features_train, features_test, labels_train, labels_test):
    clf = model
    clf.fit(features_train, labels_train.values.ravel())
    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    print("the recall for this model is :", cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
    fig = plt.figure(figsize=(6, 3)) # to plot the graph
    print("TP", cnf_matrix[1, 1, ]) # no of fraud transaction which are predicted fraud
    print("TN", cnf_matrix[0, 0]) # no. of normal transaction which are predited normal
    print("FP", cnf_matrix[0, 1]) # no of normal transaction which are predicted fraud
    print("FN", cnf_matrix[1,0 ]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))


def under_sample(data, normal_indices, fraud_indices, count_fraud_transaction, times):
    print("len", count_fraud_transaction)
    normal_indices = np.array(
        np.random.choice(normal_indices, (times * count_fraud_transaction), replace=False))
    undersample_data = np.concatenate([fraud_indices, normal_indices])
    undersample_data = data.iloc[undersample_data, :]

    total = len(normal_indices) + len(fraud_indices)
    print("len total", total)
    print("the normal transaction proportion is :",
          len(undersample_data[undersample_data.Class == 0]) / total)
    print("the fraud transaction proportion is :",
          len(undersample_data[undersample_data.Class == 1]) / total)
    print("total number of record in resampled data is:", total)
    return undersample_data


def under_sampling():

    data = load_dataset()

    count_fraud_transaction = len(data[data["Class"] == 1])

    fraud_indices = np.array(data[data.Class == 1].index)
    normal_indices = np.array(data[data.Class == 0].index)

    under_sample_data = under_sample(data, normal_indices, fraud_indices, count_fraud_transaction, 1)

    x = under_sample_data.ix[:, under_sample_data.columns != "Class"]
    y = under_sample_data.ix[:, under_sample_data.columns == "Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print("length of training data")
    print(len(x_train))
    print("length of test data")
    print(len(y_test))

    count = 0
    for classifier_name, classifier in dict_classifiers.items():

        # train with my classifier
        train_data(classifier, classifier_name, x_train, y_train, x_test, y_test)

        print()
        print("------------------------------------------------------------")
        print()

        print("Running... Cross validation with Classifier: ", classifier_name)
        start = time.time()
        scores = cross_val_score(classifier, x, y.values.ravel(), cv=10)
        stop = time.time()
        print("Cross - validated scores", scores)
        print("%20s Accuracy: %0.2f (+/- %0.2f), time:%.4f" % (
            classifier_name, scores.mean(), scores.std() * 2, stop - start))

        print()
        print("------------------------------------------------------------")
        print()

        model(classifier, x_train, x_test, y_train, y_test)

        print()
        print("------------------------------------------------------------")
        print()

        count += 1


def smote_sampling():

    data = load_dataset()

    x = data.ix[:, data.columns != "Class"]
    y = data.ix[:, data.columns == "Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    over_sampling = SMOTE(random_state=0)

    over_sampling_data_x, over_sampling_data_y = over_sampling.fit_sample(x_train, y_train.values.ravel())
    over_sampling_data_x = pd.DataFrame(data=over_sampling_data_x, columns=x_train.columns)
    over_sampling_data_y = pd.DataFrame(data=over_sampling_data_y, columns=["Class"])

    print("length of oversampled data is ", len(over_sampling_data_x))
    print("Number of normal transcation in oversampled data", len(over_sampling_data_y[over_sampling_data_y["Class"] == 0]))
    print("No.of fraud transcation", len(over_sampling_data_y[over_sampling_data_y["Class"] == 1]))

    print("Proportion of Normal data in oversampled data is ", len(over_sampling_data_y[over_sampling_data_y["Class"] == 0]) / len(over_sampling_data_x))
    print("Proportion of fraud data in oversampled data is ", len(over_sampling_data_y[over_sampling_data_y["Class"] == 1]) / len(over_sampling_data_x))

    print("length of training data")
    print(len(x_train))
    print("length of test data")
    print(len(y_test))

    count = 0
    for classifier_name, classifier in dict_classifiers.items():
        # train with my classifier
        train_data(classifier, classifier_name, x_train, y_train, x_test, y_test)

        print()
        print("------------------------------------------------------------")
        print()

        print("Running... Cross validation with Classifier: ", classifier_name)
        start = time.time()
        scores = cross_val_score(classifier, x, y.values.ravel(), cv=10)
        stop = time.time()
        print("Cross - validated scores", scores)
        print("%20s Accuracy: %0.2f (+/- %0.2f), time:%.4f" % (
            classifier_name, scores.mean(), scores.std() * 2, stop - start))

        print()
        print("------------------------------------------------------------")
        print()

        model(classifier, x_train, x_test, y_train, y_test)

        print()
        print("------------------------------------------------------------")
        print()

        count += 1


#under_sampling()
smote_sampling()