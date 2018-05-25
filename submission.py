import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helper

def Concatenate(data):
    return [ ' '.join(item) for item in data ]


def SimpleTokenize(s):
    return s.split(' ')


def construct_replace_list(classifier, feature_names):
    coef = classifier.coef_.toarray().ravel()
    indices = np.argsort(coef)
    to_replace = {}
    replacements = []

    # A negative coefficient means that the word indicates class 0.  Put all the words with
    # non-positive coefficients in the 'replacements' list, in ascending order of the
    # coefficient. This corresponds to descending order of importance, i.e. the earlier a
    # word is in the list, the more strongly it suggests class 0.
    for importance in range(len(indices)):
        word_index = indices[importance]
        if coef[word_index] > 0:
            break
        replacements.append(feature_names[word_index])
    # A positive coefficient means that the word indicates class 1. The 'to_replace' dictionary
    # maps all the class 1 words to their coefficients. It is used to decide which words to
    # remove or replace in each example.
    importance = 0
    for j in range(len(indices) -1, -1, -1):
        word_index = indices[j]
        if coef[word_index] <= 0:
            break
        to_replace[feature_names[word_index]] = importance
        importance += 1

##    with open('replacements_4.txt', 'w') as outfile:
##        for word in replacements:
##            print(word, file=outfile)
##    with open('to_replace_4.txt', 'w') as outfile:
##        for word in to_replace:
##            print(word, to_replace[word], file=outfile)

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

    # It seems silly to have to concatenate the data that's just been split, but there doesn't
    # seem to be a way to pass tokens to CountVectorizer. It requires the examples to be strings.
    training_data = Concatenate(strategy_instance.class0) + Concatenate(strategy_instance.class1)
    training_labels = [0] * len(strategy_instance.class0) + [1] *  len(strategy_instance.class1)

    count_vect = CountVectorizer(tokenizer=SimpleTokenize).fit(training_data)
    training_counts = count_vect.transform(training_data)
    tfidf_transformer = TfidfTransformer()
    training_idf = tfidf_transformer.fit_transform(training_counts)

    parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }
    classifier = strategy_instance.train_svm(parameters, training_idf, training_labels)

    to_replace, replacements = construct_replace_list(classifier, count_vect.get_feature_names())

    for lineNo in range(len(data)):
        line = data[lineNo]
        wordset = set(line)
        word_importances = []
        for word in wordset:
            if word in to_replace:
                word_importances.append((to_replace[word], word))

        # Remove the 20 words that most strongly indicate class 1.
        word_importances.sort()
        to_remove = set([ wi[1] for wi in word_importances[:20] ])
        new_line = []
        for i in range(len(line)):
            if line[i] not in to_remove:
                new_line.append(line[i])
        # If we couldn't find 20 words to remove then add
        # words until the total number of changes is 20.
        if len(to_remove) < 20:
            to_add = []
            ri = 0
            add_count = 20 - len(to_remove)
            while add_count > 0:
                while replacements[ri] in wordset:
                    ri += 1
                to_add.append(replacements[ri])
                ri += 1
                add_count -= 1
            new_line.extend(to_add)
#            print('Added words:', to_add)
#        print('Removed words:', to_remove)
        data[lineNo] = new_line
#        print(data[lineNo])

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as outfile:
        for line in data:
            print(' '.join(line), file=outfile)

    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
