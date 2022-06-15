import csv
import logging
import numpy as np
import os
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import pickle
from tqdm import tqdm


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']

def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])
    
def read_text(path, data_dir="./data/"):
    print("reading path: %s" % (data_dir + path))
    label_list = []
    clean_text_list = []
    if (
        path.startswith("ag_news")
        or path.startswith("dbpedia")
        or path.startswith("yahoo")
        or path.startswith("yelp")
    ):
        with open(data_dir + "%s.csv" % path, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in tqdm(csv_reader):
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(" ".join(text_to_tokens(text)))
    elif path.startswith('sst-2'):
        pos_list = []
        neg_list = []
        pos_num = 0
        neg_num = 0
        with open(data_dir + "%s.tsv" % path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.split('\t')
                if int(line[1]) == 0:
                    neg_list.append(line[0].lower().strip())
                    neg_num += 1
                else:
                    pos_list.append(line[0].lower().strip())
                    pos_num += 1
        text_list = pos_list + neg_list
        clean_text_list = [' '.join(text_to_tokens(s)) for s in text_list]
        label_list = [1] * pos_num + [0] * neg_num
    elif path.startswith('imdb'):
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, path + '/pos')
        neg_path = os.path.join(data_dir, path + '/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        clean_text_list = [' '.join(text_to_tokens(s)) for s in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)
    else:
        raise NotImplementedError
    return clean_text_list, label_list

def text_to_tokens(text):
    """
    Clean the raw text.
    """
    toks = word_tokenize(text)
    spliter = ['\'', '#', '!', '\"', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':', ';', '<', '=', '>', '?','@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    toks = [token for token in filter(lambda x: x not in spliter, toks)]
    return toks

def load_pretrained_embedding(task_name, max_vocab_size, data_dir="./", emb_type="counter"):
    # counter_embedding_matrix = np.load(
    #     data_dir
    #     + "aux_files/embeddings_counter_%s_%d.npy" % (task_name, max_vocab_size)
    # )
    counter_embedding_matrix = np.load(
        data_dir
        + "aux_files/embeddings_%s_%s_%d.npy" % (emb_type, task_name, max_vocab_size)
    )
    print("Counter embedding matrix: ", counter_embedding_matrix.shape)
    return counter_embedding_matrix

def text_encoder(texts, org_dic, maxlen):
    """
    Map the raw text to word id sequence.
    """
    seqs = []
    seqs_mask = []
    for text in texts:
        words = text.split(" ")
        mask = []
        for i in range(len(words)):
            words[i] = org_dic[words[i]] if words[i] in org_dic else 0
            mask.append(1)
        seqs.append(words)
        seqs_mask.append(mask)
    seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    seqs_mask = pad_sequences(
        seqs_mask, maxlen=maxlen, padding="post", truncating="post", value=0
    )
    return seqs, seqs_mask

def text_encoder_v2(texts, org_dic, maxlen):
    """
    Map the raw text to word id sequence.
    """
    seqs = []
    seqs_mask = []
    for text in texts:
        words = text.split(" ")
        mask = []
        for i in range(len(words)):
            words[i] = org_dic[words[i]] if words[i] in org_dic else 0
            if words[i] != 0:
                mask.append(1)
            else:
                mask.append(0)
        seqs.append(words)
        seqs_mask.append(mask)
    seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    seqs_mask = pad_sequences(
        seqs_mask, maxlen=maxlen, padding="post", truncating="post", value=0
    )
    return seqs, seqs_mask

def load_dictionary(task_name, max_vocab_size, data_dir="./"):
    with open(
        (data_dir + "aux_files/org_dic_%s_%d.pkl" % (task_name, max_vocab_size)), "rb") as f:
        org_dic = pickle.load(f)
    with open(
        (data_dir + "aux_files/org_inv_dic_%s_%d.pkl" % (task_name, max_vocab_size)), "rb") as f:
        org_inv_dic = pickle.load(f)
    return org_dic, org_inv_dic

def load_enc_dictionary(task_name, max_vocab_size, data_dir="./"):
    with open(
        (data_dir + "aux_files/enc_dic_%s_%d.pkl" % (task_name, max_vocab_size)), "rb") as f:
        enc_dic = pickle.load(f)
    return enc_dic

def load_dist_mat(dataset, max_vocab_size, data_dir="./"):
    dist_mat = np.load(
        (
            data_dir
            + "aux_files/small_dist_counter_%s_%d.npy" % (dataset, max_vocab_size)
        )
    )
    return dist_mat

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

num_labels_task = {
    "sst-2": 2,
    "imdb": 2,
    "cola":2, 
    "agnews": 4,
    "yahoo": 10,
    "yelp": 5
}