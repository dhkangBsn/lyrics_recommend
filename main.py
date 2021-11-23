import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('./data/발라드.csv', encoding='cp949')

def get_title_to_idx():
    return pickle.load(open("./data/ballad_title_to_idx.pkl" , 'rb'))

def get_cosine_tfidf():
    return pickle.load(open("./data/ballad_sims_cosine_tfidf.pkl", 'rb'))


def print_hi(name):
    title_to_idx = get_title_to_idx()
    title_idx = title_to_idx['생일 축하합니다, 그냥']
    print(f'current index : {title_idx}')

    cosine_tfidf = get_cosine_tfidf()
    top_ten = np.argsort(cosine_tfidf[title_idx])[::-1][1:11]
    print(top_ten)
    print(df.iloc[top_ten].title.values)

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
