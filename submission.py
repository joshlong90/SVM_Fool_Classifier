import numpy as np
import time
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import helper

print_stats = False

def load_training_data(label):
    lines = []
    filename = 'class-{}.txt'.format(label)
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines, [label] * len(lines)

########################################################################################################

def initialise_fool_vector(word_class0, word_class1):
    fool_vector = []
    for word in word_class0:
        fool_vector.append([abs(word_class0[word]), word, True])
    for word in word_class1:
        fool_vector.append([abs(word_class1[word]), word, False])
    fool_vector.sort(reverse=True)
    word_vector = [word[1] for word in fool_vector]
    boolean_vector = [word[2] for word in fool_vector]
    word_indicies = {}
    for index in range(len(word_vector)):
    	word_indicies[word_vector[index]] = index
    return word_indicies, word_vector, boolean_vector

def capture_word_dictionary(coef, feature_names, top_class_features, label):
    word_dict = {}
    for feature in top_class_features:
        word_dict[feature_names[feature]] = coef[feature]
    return word_dict

def determine_prominent_words(classifier, top_class0_features=20):
    coef = classifier.coef_.toarray().ravel()
    sorted_coef = np.argsort(coef)

    top_class0 = sorted_coef[:top_class0_features]
    weakest_class0 = coef[top_class0[-1]]

    top_class1_features = 1
    while coef[sorted_coef[-(top_class1_features+1)]] > abs(weakest_class0):
        top_class1_features += 1

    top_class1 = sorted_coef[-top_class1_features:]
    return coef, top_class0, top_class1

########################################################################################################

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

    class0train, class0labels = load_training_data(0)
    class1train, class1labels = load_training_data(1)
    training_data = class0train + class1train
    training_labels = class0labels + class1labels

    count_vect = CountVectorizer().fit(training_data)
    training_counts = count_vect.transform(training_data)
#    print(training_counts.shape)
    tfidf_transformer = TfidfTransformer()
    training_idf = tfidf_transformer.fit_transform(training_counts)

    parameters = { 'gamma' : 'auto', 'C' : 1.0, 'kernel' : 'linear', 'degree' : 2, 'coef0' : 0 }
    classifier = strategy_instance.train_svm(parameters, training_idf, training_labels)
    
########################################################################################################

    start_time = time.time()

    feature_names = count_vect.get_feature_names()
    feature_names = np.array(feature_names)

    coef, top_class0, top_class1 = determine_prominent_words(classifier, 60)

    word_class0 = capture_word_dictionary(coef, feature_names, top_class0, 0)
    word_class1 = capture_word_dictionary(coef, feature_names, top_class1, 1)
    
    # WORD_VECTOR contains a list of the most prominent words in terms of absolute coef value.
    # BOOLEAN_VECTOR contains a list of boolean values depicting whether the corresponding word \
    # in word_vector points to class0.
    # WORD_INDICIES maps each word in word_vector to it's index position in terms of overall prominence.
    word_indicies, word_vector, boolean_vector = initialise_fool_vector(word_class0, word_class1)

    total_added = 0
    total_removed = 0
    added_counts = Counter()
    removed_counts = Counter()

    for line in data:
        # mark the presence of whether a word appears in a document.
        present_words = set()
        for word in set(line):
            if word in word_indicies:
                present_words.add(word_indicies[word])

        count = 0
        index = 0
        add_words = []
        remove_words = set()
        while index < len(word_vector) and count < 20:
            # case where word points to class0
            if boolean_vector[index]:
                if not index in present_words:
                    add_words.append(word_vector[index])
                    count += 1
            # case where word points to class0
            elif index in present_words:
                remove_words.add(word_vector[index])
                count += 1

            index += 1

        filler_word = add_words[0]
        for i in range(len(line)):
            if line[i] in remove_words:
                line[i] = filler_word
        for word in add_words:
            line.append(word)
        # Record statistics
        total_added += len(add_words)
        total_removed += len(remove_words)
        for word in add_words:
            added_counts[word] += 1
        for word in remove_words:
            removed_counts[word] += 1

    if print_stats:
        elapsed_time = time.time() - start_time
        print('Replacement algorithm took {} seconds.'.format(elapsed_time))
        print()
        print('Added a total of {} words and removed a total of {} words.'.format(total_added, total_removed))
        print()
        print('Total number of additions per word:')
        for word, count in added_counts.most_common():
            print('{0:<20} {1:>4}'.format(word, count))
        print()
        print('Total number of removals per word:')
        for word, count in removed_counts.most_common():
            print('{0:<20} {1:>4}'.format(word, count))

########################################################################################################

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as outfile:
        for line in data:
            print(' '.join(line), file=outfile)

    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
