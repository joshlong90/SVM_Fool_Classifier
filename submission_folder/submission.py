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

    # A negative coefficient means that the word indicates class 0. Put all the words with
    # non-positive coefficients in the 'replacements' list, in ascending order of the
    # coefficient. This corresponds to descending order of importance, i.e. the earlier a
    # word is in the list, the more strongly it suggests class 0.
    replacements = []
    for importance in range(len(indices)):
        word_index = indices[importance]
        if coef[word_index] > 0:
            break
        replacements.append(feature_names[word_index])
    # A positive coefficient means that the word indicates class 1. The 'to_replace' dictionary
    # assigns a numeric rank to all the class 1 words to indicate how strongly they suggest
    # class 1. The lower the rank, the more important the word is, so that when we sort by rank
    # the most important words come first.
    rank = 0
    to_replace = {}
    for j in range(len(indices) -1, -1, -1):
        word_index = indices[j]
        if coef[word_index] <= 0:
            break
        to_replace[feature_names[word_index]] = rank
        rank += 1

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

    # It seems silly to have to re-concatenate the data that's just been split, only to have
    # CountVectorizer split it again. However, there doesn't seem to be a way to pass vectors
    # of tokens to CountVectorizer. It requires the examples to be strings.
    training_data = Concatenate(strategy_instance.class0) + Concatenate(strategy_instance.class1)
    training_labels = [0] * len(strategy_instance.class0) + [1] *  len(strategy_instance.class1)
    # Note that we use a custom tokenizer with CountVectorizer to prevent it from removing
    # punctuation.
    count_vect = CountVectorizer(tokenizer=SimpleTokenize).fit(training_data)
    training_counts = count_vect.transform(training_data)
    tfidf_transformer = TfidfTransformer()
    training_idf = tfidf_transformer.fit_transform(training_counts)
    # Train a linear SVM using a tf-idf representation of the training data.
    parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }
    classifier = strategy_instance.train_svm(parameters, training_idf, training_labels)
    # Use our SVM to determine the best words to remove, and possibly add, to fool the classifier.
    to_replace, replacements = construct_replace_list(classifier, count_vect.get_feature_names())

    for lineNo in range(len(data)):
        line = data[lineNo]
        wordset = set(line)
        # Look up the rank for each distinct word in the example, and construct a list of
        # (rank, word) tuples.
        word_ranks = []
        for word in wordset:
            if word in to_replace:
                word_ranks.append((to_replace[word], word))
        # Sort the list so that the words with the lowest rank, which most strongly indicate
        # class 1, are at the beginning.
        word_ranks.sort()
        # Construct a set of the 20 words that most strongly indicate class 1, and remove
        # these words from the example.
        to_remove = set([ wi[1] for wi in word_ranks[:20] ])
        new_line = []
        for i in range(len(line)):
            if line[i] not in to_remove:
                new_line.append(line[i])
        # If we couldn't find 20 words to remove then add words until the total number of changes
        # is 20. We add the words which most strongly indicate class 0.
        if len(to_remove) < 20:
            ri = 0
            for _ in range(20 - len(to_remove)):
                # Don't add a word if it's already in the example.
                while replacements[ri] in wordset:
                    ri += 1
                new_line.append(replacements[ri])
                ri += 1
        data[lineNo] = new_line

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as outfile:
        for line in data:
            print(' '.join(line), file=outfile)

    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
