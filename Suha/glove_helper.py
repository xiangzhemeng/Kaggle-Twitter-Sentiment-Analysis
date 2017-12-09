import numpy as np

def load_glove_model(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model

def average_embeddings(sentences, model, vector_length=200):
    return_matrix = []
    for sentence in sentences:
        words = sentence.split()
        word_count = float(len(words))
        temp_list = np.zeros(vector_length)
        for word in words:
            try:
                temp_list += model[word][:vector_length]
            except:
                word_count -= 1
        if word_count == 0:
            avrg_list = np.zeros(vector_length)
        else:
            avrg_list = temp_list/word_count
        return_matrix.append(avrg_list)
    return return_matrix
