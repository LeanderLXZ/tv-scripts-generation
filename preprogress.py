import os
import pickle

data_dir = './data/simpsons/moes_tavern_lines.txt'


# Load data

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    print(input_file)
    with open(input_file, "r") as f:
        data = f.read()

    return data


# Lookup Table

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


# Tokenize Punctuation

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    token_dict = {'.': '||Period||',
                  ',': '||Comma||',
                  '"': '||Quotation_Mark||',
                  ';': '||Semicolon||',
                  '!': '||Exclamation_Mark||',
                  '?': '||Question_Mark||',
                  '(': '||Left_Parentheses||',
                  ')': '||Right_Parentheses||',
                  '--': '||Dash||',
                  '\n': '||Return||'}
    return token_dict


# Preprocess all the data and save it

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    text = text[79:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

if __name__ == '__main__':
    preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)