import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helper
import mutual_information

# Attempt to fool classifier based on mutual information

def load_training_data(label):
    lines = []
    filename = 'class-{}.txt'.format(label)
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines, [label] * len(lines)


def construct_replace_list():
    mutual_info = mutual_information.MutualInformation()
    to_replace = {}
    replacements = []
    for i in range(len(mutual_info)):
        info = mutual_info[i]
        word = info[1]
        if info[2] > info[3]:
            # Class 0 work
            replacements.append(word)
        else:
            to_replace[word] = i

    return to_replace, replacements


def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    with open(test_data, 'r') as infile:
        data = [ line.strip().split(' ') for line in infile ]

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()
    parameters={}


    ##..................................#
    #
    #
    #
    ## Your implementation goes here....#
    #
    #
    #
    ##..................................#

#    class0train, class0labels = load_training_data(0)
#    class1train, class1labels = load_training_data(1)
#    training_data = class0train + class1train
#    training_labels = class0labels + class1labels

#    count_vect = CountVectorizer().fit(training_data)
#    training_counts = count_vect.transform(training_data)
#    print(training_counts.shape)
#    tfidf_transformer = TfidfTransformer()
#    training_idf = tfidf_transformer.fit_transform(training_counts)

#    parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }
#    classifier = strategy_instance.train_svm(parameters, training_idf, training_labels)

    to_replace, replacements = construct_replace_list()

    for line in data:
        wordset = set(line)
        word_importances = []
        for word in wordset:
            if word in to_replace:
                word_importances.append((to_replace[word], word))

        word_importances.sort()
        substitutions = {}

        count = 0
        ri = 0
        for w in word_importances:
            word = w[1]
            # Don't use a replacement word that's already in the text.
            while replacements[ri] in wordset:
                ri += 1
            substitutions[word] = replacements[ri]
            count += 1
            if count == 10:
                break
            ri += 1

#        print('Substitutions:', substitutions)
        for i in range(len(line)):
            if line[i] in substitutions:
#                print('Changing {} to {}'.format(line[i], substitutions[line[i]]))
                line[i] = substitutions[line[i]]


    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as outfile:
        for line in data:
            print(' '.join(line), file=outfile)

    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
