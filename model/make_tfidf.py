import MeCab
import numpy as np
import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import pickle

m = MeCab.Tagger()

target_tags = [
    'NNG',  # 일반 명사
    'NNP',  # 고유 명사
    'NNB',  # 의존 명사
    'NR',  # 수사
    'NP',  # 대명사
    'VV',  # 동사
    'VA',  # 형용사
    'MAG',  # 일반 부사
    'MAJ',  # 접속 부사
]


def parse_sentence(sentence, target_tags, stop_word):
    result = m.parse(sentence)
    temp = result.split('\n')
    temp_2 = [ sentence.split('\t') for sentence in temp]
    words = [ sentence[0] for sentence in temp_2 ]
    morphs = [ sentence[1].split(',')[0]
               for sentence in temp_2
               if len(sentence) > 1]
    morphs = [ morph for morph in morphs if morph in target_tags ]
    words = words[:len(morphs)]



    word_morph = [ (word,morph)
                   for morph, word in zip(morphs, words)
                   if word not in stop_word ]
    return word_morph


def extract_word_list(lyrics, target_tags, stop_word):
    result = []
    try:
        for idx in range(len(lyrics)):
            word_morph_list = parse_sentence(lyrics[idx], target_tags, stop_word)
            word = [ word_morph[0] for word_morph in word_morph_list if len(word_morph[0]) > 1]
            result.append(word)
    except:
        print(idx, '해당 인덱스에서 오류가 났습니다.')
    return result


df = pd.read_csv('../data/발라드.csv', encoding='cp949')
print(df.head())
lyrics = df['lyrics'].values
titles = df['title'].values
title_to_idx = { title:idx for idx, title in enumerate(titles) }

f = open("../data/ballad_title_to_idx.pkl", "wb")
pickle.dump(title_to_idx, f)
f.close()
print('pickle title_to_idx')
print(pickle.load(open("../data/ballad_title_to_idx.pkl" , 'rb')))



stop_word = ['것', '을', '겠', '은', '.', '는', ',']
word = extract_word_list(lyrics, target_tags, stop_word)

def make_bigram(word):
    return gensim.models.Phrases(word, min_count=5, threshold=100)

def make_trigram(word):
    bigram = gensim.models.Phrases(word, min_count=5, threshold=100)
    return gensim.models.Phrases(bigram[word], threshold=100)

#print(bigram)
def make_trigram_list(word, bigram_mod, trigram_mod):
    trigram_list = []
    for idx in range(len(word)):
        trigram_list.append(trigram_mod[bigram_mod[word[idx]]])
    return trigram_list

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = make_bigram(word)
trigram_mod = make_trigram(word)
trigram_list = make_trigram_list(word, bigram_mod, trigram_mod)
print(trigram_list[:10])
print(' '.join(trigram_list[0]))
lyrics = [' '.join(trigram) for trigram in trigram_list]
print(lyrics[0])


### TODO
# use CountVectorizer & TfidfVectorizer to construct dtm & dtm_tfidf
count = CountVectorizer(binary=False) # binary x, tf
tfidf = TfidfVectorizer()

dtm: np.ndarray = count.fit_transform(lyrics).toarray()
dtm_tfidf: np.ndarray = tfidf.fit_transform(lyrics).toarray()  # from csr_matrix to numpy array.  # 이제는 이거 한줄로 끝내기!
###############
#print(dtm)  # should be csr sparse matrix
#print(dtm.shape)  # (num_docs, num_terms) 당연히.. 대부분은 0 이겠지! - 다시한번 sparsity를 확인할 수 있다.
#print(dtm_tfidf)  # should be csr sparse matrix
#print(dtm_tfidf.shape)  # (num_docs, num_terms) 당연히.. 대부분은 0 이겠지! - 다시한번 sparsity를 확인할 수 있다.


# cosine distance -> this is vectorized...
sims_cosine = cosine_similarity(dtm)
sims_cosine_tfidf = cosine_similarity(dtm_tfidf)  # this may take a while...
print('sims_cosin_tfidf', sims_cosine_tfidf)



f = open("../data/ballad_sims_cosine_tfidf.pkl", "wb")
pickle.dump(sims_cosine_tfidf, f)
f.close()
print('pickle')
print(pickle.load(open("../data/ballad_sims_cosine_tfidf.pkl" , 'rb')))



target_title = '이 못난 나를 (Prod. By LA박피디-박상균)'
target_idx = title_to_idx[target_title]
print(sims_cosine_tfidf)
print(sims_cosine.shape)  # (num_doc, num_doc)
print(sims_cosine_tfidf.shape)  # (num_doc, num_doc)
print(np.sort(sims_cosine_tfidf[target_idx, :])[::-1][:10])
recommend_idx_list = np.argsort(sims_cosine_tfidf[target_idx, :])[::-1][:10]
print(recommend_idx_list)
print(df.iloc[recommend_idx_list])


# Note that these are distances, now
dists_manhattan_tfidf = manhattan_distances(dtm_tfidf)
#print(dists_manhattan_tfidf.shape)  # (num_doc, num_doc)
#print(np.sort(dists_manhattan_tfidf[0, :])[:10])
#print(np.argsort(dists_manhattan_tfidf[0, :])[:10])
#print(lyrics[0])
#print(lyrics[189])


dists_euclidean_tfidf = euclidean_distances(dtm_tfidf)  # this may take a while... not efficient at all? cosine - it didn't take that much of a time. but manhattan... well that took some time.
#print(dists_euclidean_tfidf.shape)  # (num_doc, num_doc)
#print(np.sort(dists_euclidean_tfidf[0, :])[:10])
#print(np.argsort(dists_euclidean_tfidf[0, :])[:10])
#print(lyrics[0])
#print(lyrics[299])
#print(lyrics[890])