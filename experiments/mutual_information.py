from collections import Counter
from math import log

# Count of how many examples each word appears in

def DocumentCounts(filename):
    document_counts = Counter()
    total_documents = 1
    with open(filename,'r') as file:
        for line in file:
            unique_words = set(line.strip().split(' '))
            for word in unique_words:
                document_counts[word] += 1
            total_documents += 1
            if total_documents == 180:
                break
    
    return total_documents, document_counts

class_0_docs, class0_doc_counts = DocumentCounts('class-0.txt')

class_1_docs, class1_doc_counts = DocumentCounts('class-1.txt')

N = class_0_docs + class_1_docs

all_words = set(class0_doc_counts.keys()).union(set(class1_doc_counts.keys()))

mutual_info = []

for word in all_words:
    N_10 = class0_doc_counts[word] + 1
    N_11 = class1_doc_counts[word] + 1

    N_00 = class_0_docs - N_10
    N_01 = class_1_docs - N_11

    N_0_ = N_00 + N_01
    N_1_ = N_10 + N_11
    N__0 = N_00 + N_10
    N__1 = N_01 + N_11

    if N_00 > 0 or N_01 > 0:
        info = ((N_11 * log((N * N_11) / (N_1_ * N__1), 2))
             + (N_01 * log((N * N_01) / (N_0_ * N__1), 2))
             + (N_10 * log((N * N_10) / (N_1_ * N__0), 2))
             + (N_00 * log((N * N_00) / (N_0_ * N__0), 2))) / N
        mutual_info.append((info, word, class0_doc_counts[word] / class_0_docs, class1_doc_counts[word] / class_1_docs))

mutual_info.sort(reverse=True)

for i in range(20):
    info = mutual_info[i]
    word = info[1]
    print(info[0], word, info[2], info[3])

print(len(all_words))
