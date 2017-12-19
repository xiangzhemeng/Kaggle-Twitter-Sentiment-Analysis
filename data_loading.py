import pandas as pd

def load_data(filename):
    data = []
    for line in open(filename, "r"):
        data.append(line)
    return data


def gen_data():
    print(" == Loading Original step ==")

    pos_train_tweets = pd.DataFrame(load_data('data/tweets/train_pos_full.txt'), columns=['tweet'])
    neg_train_tweets = pd.DataFrame(load_data('data/tweets/train_neg_full.txt'), columns=['tweet'])
    train_tweets = pd.concat([pos_train_tweets, neg_train_tweets], axis=0)
    train_tweets.to_pickle('data/pickles/train_origin.pkl')

    test_tweets = pd.DataFrame(load_data('data/tweets/test_data.txt'), columns=['tweet'])
    test_tweets['tweet'] = test_tweets['tweet'].apply(lambda tweet: tweet.split(',', 1)[-1])
    test_tweets.to_pickle('data/pickles/test_origin.pkl')



if __name__ == "__main__":
    gen_data()