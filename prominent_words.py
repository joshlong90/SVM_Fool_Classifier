from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

################################################################################################
import submission
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
    #print('Correctly classified {} out of {} class {} examples ({}%) in {}.'.format(
    #    correct, len(examples), real_class, round(100 * correct / len(examples), 1), filename))
    if correct > 0:
        return False
    return True

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

kernel = 'poly'
coef0 = 100
degree = 2
gamma = 0.0001220703125
C = 32

parameters = { 'gamma' : 0.0001220703125, 'C' : 32, 'kernel' : 'poly', 'degree' : 2, 'coef0' : 100 }

clf = train_our_svm(parameters, training_idf, training_labels)

#print(clf)

classify_file('class-0.txt', 0, count_vect, tfidf_transformer, clf)

classify_file('test_data.txt', 1, count_vect, tfidf_transformer, clf)

submission.fool_classifier('test_data.txt')

classify_file('modified_data.txt', 1, count_vect, tfidf_transformer, clf)

def example_create():
    word_count0 = {'iraq': -1.3428719720634539, 'of': -1.2931977466056304, 'more': -1.087766429439081, 'be': -0.9334012972832332, 'city': -0.8863224943158301, 'three': -0.8862499580632779, 'bomb': -0.8396715420942247, 'the': -0.8233782791047415, 'nuclear': -0.7961775218152379, 'himself': -0.7948641578258017, '1996': -0.7542242016571657, 'than': -0.7469820167589973, 'force': -0.7382600726318713, 'fail': -0.7338137953756454, 'weapon': -0.6856920283554442, 'east': -0.6851369332124994, 'division': -0.6626174871179022, '12': -0.6477959702078969, 'kurdish': -0.6451651588179804, 'most': -0.6358177768436739, 'embassy': -0.6348200788312363, 'international': -0.6292899234209195, 'move': -0.6283637767300021, 'it': -0.6262588721082037, '08': -0.6246440594344719, 'not': -0.6213309665307166, 'zimbabwe': -0.6075232333099865, 'percent': -0.602647107359328, 'oil': -0.600608393192702, 'kashmir': -0.5963863612190949, 'afghanistan': -0.5937008897047711, 'start': -0.5881774141606234, 'iran': -0.5826092418889962, 'struggle': -0.5757425417324435, 'however': -0.5713382889869456, 'peninsula': -0.5670646136527201, 'national': -0.5667720735865349, 'sea': -0.5645366956111071, 'rise': -0.5645229740881109, 'into': -0.5636891643276973, 'television': -0.5614958343994279, 'get': -0.5553620118141647, 'through': -0.5546672983695279, 'bank': -0.554139753800776, '17': -0.5521190054118688, '25': -0.5512942816075749, 'support': -0.5511856613680548, 'sanction': -0.5508464508656874}
    count = 0
    solutions = []
    word = ""
    for i in word_count0:
        count = 0
        for j in word_count0:       
            for k in word_count0:
                example = ["clash", "president"]
                example = example + [i, j, k]
                with open("test_1.txt", 'w') as outfile:
                    print(' '.join(example), file=outfile)
                if classify_file('test_1.txt', 1, count_vect, tfidf_transformer, clf):
                    count += 1
        
        solutions.append([count, i])
        solutions.sort()
        print(solutions)


example_create()



#plot_coefficients(clf, count_vect.get_feature_names(), 40)
