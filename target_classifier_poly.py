from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

import submission

import matplotlib.pyplot as plt

def train_our_svm(parameters, x_train, y_train):
    ## Populate the parameters...
    gamma=parameters['gamma']
    C=parameters['C']
    kernel=parameters['kernel']
    degree=parameters['degree']
    coef0=parameters['coef0']

    ## Train the classifier...
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
#    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    return clf

def load_training_data(filenames, label, i, j):
    lines = []
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())
    return lines, [label] * len(lines)

def classify_examples(examples, real_class, count_vect, tfidf_transformer, classifier):
    counts = count_vect.transform(examples)
    tfidf = tfidf_transformer.transform(counts)
#    print(tfidf.shape)
    predictions = classifier.predict(tfidf)
    correct = 0
    for predicted_class in predictions:
        if predicted_class == real_class:
            correct += 1
    return correct

def classify_file(examples, real_class, count_vect, tfidf_transformer, classifier):
    correct = classify_examples(examples, real_class, count_vect, tfidf_transformer, classifier)
    print('Correctly classified {} out of {} class {} examples ({}%) in {}.'.format(
        correct, len(examples), real_class, round(100 * correct / len(examples), 1)))

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.toarray().ravel()
#    coef = classifier.dual_coef_.toarray().ravel()

    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = [ 'red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

# Deliberately train an over-fitted classifier

with open('class-0.txt','r') as class0file:
    class0 = [line.strip() for line in class0file]
    class0train = class0[0:180]
    class0test = class0[180:]

with open('class-1.txt','r') as class1file:
    class1train = [line.strip() for line in class1file]

with open('test_data.txt', 'r') as testfile:
    class1test = [ line.strip() for line in testfile ]

with open('modified_data.txt', 'r') as modifiedfile:
    modified_data = [ line.strip() for line in modifiedfile ]


class0labels = [0]*len(class0train)
class1labels = [1]*len(class1train)

training_data = class0train + class1train
training_labels = class0labels + class1labels



count_vect = CountVectorizer().fit(training_data)

training_counts = count_vect.transform(training_data)

tfidf_transformer = TfidfTransformer().fit(training_counts)
training_idf = tfidf_transformer.transform(training_counts)
 
best_score = 0


kernel = 'poly'
coef0 = 100
degree = 2
gamma = 0.0001220703125
C = 32
print("gamma = " + str(gamma) + ", C = " + str(C))
parameters = { 'gamma' : gamma, 'C' : C, 'kernel' : kernel, 'degree' : degree, 'coef0' : coef0 }
clf = train_our_svm(parameters, training_idf, training_labels)

correct1 = classify_examples(class0test, 0, count_vect, tfidf_transformer, clf)
print('Correctly classified {} out of {} class {} examples.'.format(
correct1, len(class0test), 0))

correct2 = classify_examples(class1test, 1, count_vect, tfidf_transformer, clf)
print('Correctly classified {} out of {} class {} examples.'.format(
correct2, len(class1test), 1))

submission.fool_classifier('test_data.txt')
correct = classify_examples(modified_data, 1, count_vect, tfidf_transformer, clf)
print('Correctly classified {} out of {} class {} examples.'.format(
correct, len(modified_data), 1))
print()

for coef0 in [0, 1, 4, 10, 100]:
    for degree in [1, 2, 3, 4, 5, 10]:
        for gamma in [2**x for x in range(-15, 4)]:
            for C in [2**x for x in range(-5, 16, 2)]:
                print("gamma = " + str(gamma) + ", C = " + str(C))
                parameters = { 'gamma' : gamma, 'C' : C, 'kernel' : kernel, 'degree' : degree, 'coef0' : coef0 }
                clf = train_our_svm(parameters, training_idf, training_labels)

                correct1 = classify_examples(class0test, 0, count_vect, tfidf_transformer, clf)
                print('Correctly classified {} out of {} class {} examples.'.format(
                correct1, len(class0test), 0))

                correct2 = classify_examples(class1test, 1, count_vect, tfidf_transformer, clf)
                print('Correctly classified {} out of {} class {} examples.'.format(
                correct2, len(class1test), 1))

                submission.fool_classifier('test_data.txt')
                correct = classify_examples(modified_data, 1, count_vect, tfidf_transformer, clf)
                print('Correctly classified {} out of {} class {} examples.'.format(
                correct, len(modified_data), 1))
                print()

                if correct1 + correct2 > best_score:
                    best_score = correct1 + correct2
                    best_parameters = [kernel, coef0, degree, gamma, C]
print(best_parameters, best_score)

