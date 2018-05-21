from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

################################################################################################
import submission as submission
################################################################################################

import matplotlib.pyplot as plt

overfit = False
polynomial = False

def train_our_svm(parameters, x_train, y_train):
    ## Populate the parameters...
    gamma=parameters['gamma']
    C=parameters['C']
    kernel=parameters['kernel']
    degree=parameters['degree']
    coef0=parameters['coef0']

    ## Train the classifier...
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
    clf.fit(x_train, y_train)
    return clf

def load_training_data(filenames, label):
    lines = []
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())
    return lines

def classify_examples(filename, examples, real_class, tfidf_vectorizer, classifier):
    tfidf = tfidf_vectorizer.transform(examples)
#    print(tfidf.shape)
    predictions = classifier.predict(tfidf)
    correct = 0
    for predicted_class in predictions:
        if predicted_class == real_class:
            correct += 1
    print('Correctly classified {} out of {} class {} examples ({}%) in {}.'.format(
        correct, len(examples), real_class, round(100 * correct / len(examples), 1), filename))
    return correct

def classify_file(filename, real_class, tfidf_vectorizer, classifier):
    with open(filename, 'r') as file:
        examples = [line.strip() for line in file]
    correct = classify_examples(filename, examples, real_class, tfidf_vectorizer, classifier)

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

if overfit:
    # Deliberately train an over-fitted classifier by training on the test data too.
    class0train = load_training_data(['class-0.txt'], 0)
    class0test = class0train
    class1train = load_training_data(['class-1.txt', 'test_data.txt'], 1)
    print('Overfitting classifier.')
else:
    with open('class-0.txt','r') as class0file:
        class0 = [line.strip() for line in class0file]
        class0train = class0[0:180]
        class0test = class0[180:]
    class1train = load_training_data(['class-1.txt'], 1)
    print('Not overfitting classifier.')

training_data = class0train + class1train
training_labels = [0] * len(class0train) + [1] * len(class1train)

#count_vect = CountVectorizer().fit(training_data)

#training_counts = count_vect.transform(training_data)

#print(training_counts.shape)

tfidf_vectorizer = TfidfVectorizer(binary=True).fit(training_data)
training_idf = tfidf_vectorizer.transform(training_data)

#print(training_idf.shape)
#print(type(training_idf))

if polynomial:
    parameters = { 'gamma' : 0.0001220703125, 'C' : 32, 'kernel' : 'poly', 'degree' : 2, 'coef0' : 1000 }
else:
    parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }

clf = train_our_svm(parameters, training_idf, training_labels)

#print(clf)

classify_examples('class-0.txt', class0test, 0, tfidf_vectorizer, clf)

classify_file('test_data.txt', 1, tfidf_vectorizer, clf)

submission.fool_classifier('test_data.txt')

classify_file('modified_data.txt', 1, tfidf_vectorizer, clf)

if not polynomial:
    plot_coefficients(clf, tfidf_vectorizer.get_feature_names(), 40)
