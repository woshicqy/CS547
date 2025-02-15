import numpy as np
import os
import nltk
import itertools
import io

nltk.download('punkt')
def get_all_reviews(directory, load_train_data=True):
    """Helper function to get all the reviews needed by 
    preprocess_review and preprocess_glove_features"""

    pos_filenames = os.listdir(directory + 'pos/')
    neg_filenames = os.listdir(directory + 'neg/')

    pos_filenames = [directory+'pos/'+filename for filename in pos_filenames]
    neg_filenames = [directory+'neg/'+filename for filename in neg_filenames]

    if load_train_data:
        # Including unlabeled training reviews
        unsup_filenames = os.listdir(directory + 'unsup/')
        unsup_filenames = [directory+'unsup/'+filename for filename in unsup_filenames]
        filenames = pos_filenames + neg_filenames + unsup_filenames
    else:
        filenames = pos_filenames + neg_filenames

    count = 0
    reviews = []
    for filename in filenames:
        with io.open(filename,'r',encoding='utf-8') as f:
            line = f.readlines()[0]
        line = line.replace('<br />',' ')
        line = line.replace('\x96',' ')
        line = nltk.word_tokenize(line)
        line = [w.lower() for w in line]

        reviews.append(line)
        count += 1
        # print(count)
    
    return reviews
    
def preprocess_reviews(x_train, x_test):
    # Create directory to store preprocessed data
    if not os.path.isdir('preprocessed_data'):
        os.mkdir('preprocessed_data')

    # Number of tokens per review
    no_of_tokens = []
    for tokens in x_train:
        no_of_tokens.append(len(tokens))
    no_of_tokens = np.asarray(no_of_tokens)
    print("==> Number of tokens")
    print('Total: ', np.sum(no_of_tokens), 
          ' Min: ', np.min(no_of_tokens), 
          ' Max: ', np.max(no_of_tokens), 
          ' Mean: ', np.mean(no_of_tokens), 
          ' Std: ', np.std(no_of_tokens))
    
    # Associate an id to every unique token in the training data
    all_tokens = itertools.chain.from_iterable(x_train)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

    all_tokens = itertools.chain.from_iterable(x_train)
    id_to_word = [token for idx, token in enumerate(set(all_tokens))]
    id_to_word = np.asarray(id_to_word)

    # Sort the indices by word frequency instead of random
    x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
    count = np.zeros(id_to_word.shape)
    for x in x_train_token_ids:
        for token in x:
            count[token] += 1
    indices = np.argsort(-count)
    id_to_word = id_to_word[indices]
    count = count[indices]

    hist = np.histogram(count,bins=[1,10,100,1000,10000])
    print("==> Token frequencies")
    print(hist)
    for i in range(10):
        print(id_to_word[i], count[i])

    # Recreate word_to_id based on sorted list
    word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

    # Assign -1 if token doesn't appear in our dictionary
    # add +1 to all token ids, we went to reserve id=0 for an unknown token
    x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

    # Save dictionary
    np.save('preprocessed_data/imdb_dictionary.npy',np.asarray(id_to_word))

    # Save training data to single text file
    with io.open('preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

    # Save test data to single text file
    with io.open('preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")
        
def preprocess_glove_features(glove_filename, x_train, x_test):
    # glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
    with io.open(glove_filename,'r',encoding='utf-8') as f:
        lines = f.readlines()

    glove_dictionary = []
    glove_embeddings = []
    count = 0
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        glove_dictionary.append(line[0])
        embedding = np.asarray(line[1:],dtype=np.float)
        glove_embeddings.append(embedding)
        count+=1
        if(count>=100000):
            break

    glove_dictionary = np.asarray(glove_dictionary)
    glove_embeddings = np.asarray(glove_embeddings)
    # Added a vector of zeros for the unknown tokens
    glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

    word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

    x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

    np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
    np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

    with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

    with io.open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

if __name__ == "__main__":
    # Get all of the training and testing reviews
    x_train = get_all_reviews("aclImdb/train/")
    x_test = get_all_reviews("aclImdb/test/", load_train_data=False)

    # preprocess_reviews(x_train, x_test)
    preprocess_glove_features("glove/glove.840B.300d.txt", x_train, x_test)