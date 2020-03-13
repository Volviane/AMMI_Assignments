import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))

    return  data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    dot_prod = u@v.T
    cos_sim = dot_prod/(norm_u * norm_v)
    return cos_sim

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = None
    ## FILL CODE
    for word in word_vectors:
        if word not in exclude_words:
            sim = cosine(x, word_vectors[word])
            if sim > best_score :
                best_score = sim
                best_word = word

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []

    ## FILL CODE
    for word in vectors:
        sim = cosine(x, vectors[word])
        if len(heap) < k:
            heappush(heap , (sim, word))
        else :
            heappushpop(heap, (sim, word))

    return [heappop(heap) for i in range(len(heap))][::-1]

## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    a = a.lower()
    b = b.lower()
    c = c.lower()
    
    v_a = word_vectors[a]
    v_b = word_vectors[b]
    v_c = word_vectors[c]

    x = v_b - v_a + v_c
    return nearest_neighbor(x, word_vectors, exclude_words=[a,b,c])
    
    # n_a = v_a/np.linalg.norm(v_a)
    # n_b = v_b/np.linalg.norm(v_b)
    # n_c = v_c/np.linalg.norm(v_c)
    
    
    # #norm = np.linalg.norm(v_b - v_a + v_c)
    # analogie = ''
    # best_score = float('-inf')
    # for word in word_vectors:
        
    #     if True in [i in word for i in [a, b,c] ]: #word not in [a,b,c]:
    #         continue
    #     n_word = word_vectors[word]/np.linalg.norm(word_vectors[word])
    #     anal = (n_c + n_b - n_a)@n_word.T
    #     #anal = ((v_b - v_a + v_c)@word_vectors[word].T)/norm
    #     if anal > best_score:
    #         best_score = anal
    #         analogie = word
    # return analogie

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = 0.0
    ## FILL CODE
    card_A = len(A)
    card_B = len(B)
    sum_A = 0.0
    sum_B = 0.0
    
    for a in A:
        sum_A += cosine(vectors[w], vectors[a])
    sum_A /= card_A
    
    for b in B:
        sum_B += cosine(vectors[w], vectors[b])
    sum_B /= card_B
    
    strength = sum_A - sum_B 
    return strength

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = 0.0
    ## FILL CODE
    weat_X = 0.0
    weat_Y = 0.0
    for x in X:
        weat_X += association_strength(x, A, B, vectors)
    for y in Y:
        weat_Y += association_strength(y, A, B, vectors)
    
        
    score = weat_X - weat_Y
    return score

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors(sys.argv[1])


print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print (word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
