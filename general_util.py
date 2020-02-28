import numpy as np
import torch
import os

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings
    :param glove_filepath(str): path to the glove embeddings file
    :return word_to_index(dict), embeddings(numpy.ndarray):
    """
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)

    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, words):
    """ Create embedding matrix for a specific set of words

    :param glove_filepath (str): file path to the glove embeddings
    :param words (list): list of words in the dataset
    :return:
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings

def pos(x):
    '''
    map the relative distance between [0, 123)
    '''
    if x < -60:
        return 0
    if 60 >= x >= -60:
        return x + 61
    if x > 60:
        return 122

def createWin(l_input, win_size):
    n =win_size
    result = []
    for i in range(len(l_input) - n + 1):
        ngramTmp = l_input[i:i+n]
        result.append(ngramTmp)
    return result

if __name__ == '__main__':
    l_input = [i for i in range(7)]
    result=createWin(l_input, win_size=3)
    print(result)