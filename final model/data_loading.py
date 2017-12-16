import pandas as pd

def load_data(filename):
    data = []
    for line in open(filename, "r"):
        data.append(line)
    return data

pos_train_tweets = pd.DataFrame(load_data('twitter-datasets/train_pos_full.txt'), columns=['tweet'])
neg_train_tweets = pd.DataFrame(load_data('twitter-datasets/train_neg_full.txt'), columns=['tweet'])
train_tweets = pd.concat([pos_train_tweets, neg_train_tweets], axis=0)
train_tweets.to_pickle('origin_data/train_origin.pkl')

test_tweets = pd.DataFrame(load_data('twitter-datasets/test_data.txt'), columns=['tweet'])
test_tweets['tweet'] = test_tweets['tweet'].apply(lambda tweet: tweet.split(',', 1)[-1])
test_tweets.to_pickle('origin_data/test_origin.pkl')