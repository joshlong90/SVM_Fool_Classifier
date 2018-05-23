from sklearn import svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


import numpy as np

def load_training_data(filenames, label):
    lines = []
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())
    return lines

def classify_examples(tfidf, labels, classifier):
    predictions = classifier.predict(tfidf)
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    print('Correctly classified {} out of {} examples ({}%).'.format(correct, len(examples), round(100 * correct / len(examples), 1)))
    return correct

with open('class-0.txt','r') as class0file:
    class0 = [line.strip() for line in class0file]
    class0train = class0[0:180]
    class0test = class0[180:]

class1train = load_training_data(['class-1.txt'], 1)
class1test = load_training_data(['test_data.txt'], 1)
class1test = class1test[180:]

training_data = class0train + class1train
training_labels = [0] * len(class0train) + [1] * len(class1train)

test_data = class0test + class1test
test_labels = [0] * len(class0test) + [1] * len(class1test)

count_vect = CountVectorizer(ngram_range=(1, 1)).fit(training_data)
training_counts = count_vect.transform(training_data)
tfidf_transformer = TfidfTransformer().fit(training_counts)
training_idf = tfidf_transformer.transform(training_counts)

test_counts = count_vect.transform(test_data)
test_idf = tfidf_transformer.transform(test_counts)

#print(training_idf.shape)
#print(type(training_idf))

parameters = [{ 'kernel' : ['linear'], 'C' : [1, 10, 32, 100, 1000] },
              { 'kernel' : ['poly'], 'gamma' : [1e-4, 1e-3, 0.0001220703125], 'C' : [1, 10, 32, 100, 1000], 'degree' : [2, 3, 4], 'coef0' : [10, 50, 100, 500, 1000] }]

clf = GridSearchCV(svm.SVC(), parameters, cv=5, scoring=metrics.make_scorer(metrics.accuracy_score, greater_is_better=True))
clf.fit(training_idf, training_labels)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

#print(clf)

#classify_examples(test_idf, test_labels, clf)
