from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

################################################################################################
import submission_2 as submission
################################################################################################

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

def load_training_data(filenames, label):
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

def classify_file(filename, real_class, count_vect, tfidf_transformer, classifier):
    with open(filename, 'r') as file:
        examples = [line.strip() for line in file]
    correct = classify_examples(examples, real_class, count_vect, tfidf_transformer, classifier)
    print('Correctly classified {} out of {} class {} examples ({}%) in {}.'.format(
        correct, len(examples), real_class, round(100 * correct / len(examples), 1), filename))

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

class0train, class0labels = load_training_data(['class-0.txt'], 0)
class1train, class1labels = load_training_data(['class-1.txt', 'test_data.txt'], 1)

training_data = class0train + class1train
training_labels = class0labels + class1labels

count_vect = CountVectorizer().fit(training_data)

training_counts = count_vect.transform(training_data)

#print(training_counts.shape)

tfidf_transformer = TfidfTransformer().fit(training_counts)
training_idf = tfidf_transformer.transform(training_counts)

#print(training_idf.shape)
#print(type(training_idf))

parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }

clf = train_our_svm(parameters, training_idf, training_labels)

#print(clf)

classify_file('class-0.txt', 0, count_vect, tfidf_transformer, clf)

classify_file('test_data.txt', 1, count_vect, tfidf_transformer, clf)

submission.fool_classifier('test_data.txt')

classify_file('modified_data.txt', 1, count_vect, tfidf_transformer, clf)

plot_coefficients(clf, count_vect.get_feature_names(), 40)
