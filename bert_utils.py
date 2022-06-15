import csv
import logging
import sys
import numpy as np
import os
import random
import string
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import io
import torch
import re
from tqdm import tqdm


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, flaw_labels=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.flaw_labels = flaw_labels

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = example.label
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def convert_examples_to_features_v2(examples, max_seq_length, tokenizer, vocab):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        pgd_mask = []
        for token in tokens:
            if token in vocab:
                pgd_mask.append(1)
            else:
                pgd_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        pgd_mask += padding

        label_id = example.label
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures_v2(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              pgd_mask=pgd_mask))
    return features

def convert_examples_to_features_tpl(examples, max_seq_length, tokenizer, vocab):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        assert(len(tokens_a) == len(example.text_a.split()))

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)


        tokens_c = example.text_a.split()
        if len(tokens_c) > max_seq_length - 2:
            tokens_c = tokens_c[:(max_seq_length - 2)]
        tokens_c = ["[CLS]"] + tokens_c + ["[SEP]"]
        attack_ids = [vocab[w] if w in vocab else 0 for w in tokens_c]

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        attack_ids += padding

        label_id = example.label
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures_tpl(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              attack_ids=attack_ids))
    return features




class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(line)#list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class IMDBProcessor(DataProcessor):
    """Processor for the IMDB-Binary data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_test.csv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_outputs.csv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "gnrt_outputs.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            flaw_labels = None
            # spliter = re.split(r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])", line[1].lower())
            # clean_tokens = [token for token in filter(lambda x:(x != '' and x != ' '), spliter)]
            tokens = word_tokenize(line[1].lower())
            spliter = ['\'', '#', '!', '\"', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':', ';', '<', '=', '>', '?','@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
            clean_tokens = [token for token in filter(lambda x: x not in spliter, tokens)]
            text_a = ' '.join(clean_tokens)
            label = line[0]
            if len(line) == 3: flaw_labels = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples

class AGNewsProcessor(DataProcessor):
    """Processor for the AG's News data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_test.csv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_outputs.csv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "gnrt_outputs.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            flaw_labels = None
            spliter = re.split(r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])", ' . '.join(line[1:]).lower())
            clean_tokens = [token for token in filter(lambda x:(x != '' and x != ' '), spliter)]
            text_a = ' '.join(clean_tokens)
            label = str(int(line[0])-1)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples


class YahooProcessor(DataProcessor):
    """Processor for the Yahoo! Answers data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_test.csv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_csv(os.path.join(data_dir, "disc_outputs.csv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "gnrt_outputs.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            flaw_labels = None
            spliter = re.split(r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])", ' . '.join(line[1:]).lower())
            clean_tokens = [token for token in filter(lambda x:(x != '' and x != ' '), spliter)]
            text_a = ' '.join(clean_tokens)
            label = str(int(line[0])-1)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples

class SST2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "disc_dev.tsv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "disc_outputs.tsv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "gnrt_outputs.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            flaw_labels = None
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if len(line) == 3: flaw_labels = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples

def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])

def get_adv_examples_from_file(adv_fn):
    adv_examples = []
    ori_examples = []
    ori_labels = []
    adv_labels = []
    with open(adv_fn, "r", encoding="utf-8") as f:
        logger.info("Reading adversarial examples from %s", adv_fn)
        for line in f.readlines():
            line = line.strip("\n")
            if line.startswith("original text: "):
                ori_examples.append(line.replace("original text: ", "", 1))
            elif line.startswith("original label: "):
                ori_labels.append(int(line.replace("original label: ", "", 1)))
            if line.startswith("perturbed text: "):
                adv_examples.append(line.replace("perturbed text: ", "", 1))
            elif line.startswith("perturbed label: "):
                adv_labels.append(int(line.replace("perturbed label: ", "", 1)))
            else:
                continue
        logger.info("Num of loaded original examples: %d", len(ori_examples))
        logger.info("Num of loaded adversarial examples: %d", len(adv_examples))
    return ori_examples, adv_examples, ori_labels, adv_labels

def read_text(path, data_dir="./data/", rob_num=None):
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
            for row in csv_reader:
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(" ".join(text_to_tokens(text)))
                if rob_num is not None and count >= rob_num:
                    break
    elif path.startswith('imdb'):
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, path + '/pos')
        neg_path = os.path.join(data_dir, path + '/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        if rob_num:
            pos_files = sorted(pos_files, key=lambda x : helper_name(x))[:int(rob_num/2)]
            neg_files = sorted(neg_files, key=lambda x : helper_name(x))[:int(rob_num/2)]
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
    # spliter = re.split(r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])", text.lower())
    # clean_tokens = [token for token in filter(lambda x:(x != '' and x != ' '), spliter)]
    # return clean_tokens

def get_examples_for_bert(path, data_dir="./data/", aug_data_fn=None, adv_ratio=0.2):
    examples = []
    clean_text_list, label_list = read_text(path, data_dir)
    if aug_data_fn and "train" in path:
        _, aug_text_list, aug_label_list, _ = get_adv_examples_from_file(aug_data_fn)
        assert(len(clean_text_list) * adv_ratio == len(aug_text_list))
        clean_text_list = clean_text_list + aug_text_list
        label_list = label_list + aug_label_list
    for i, (text, label) in enumerate(zip(clean_text_list, label_list)):
        guid = "%s" % i
        flaw_labels = None
        examples.append(
            InputExample(guid=guid, text_a=text, text_b=None, label=label, flaw_labels=flaw_labels)
        )
    return examples

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

processors = {
    "sst-2": SST2Processor,
    "imdb": IMDBProcessor,
    "cola": SST2Processor,
    "agnews": AGNewsProcessor,
    "yahoo": YahooProcessor,
}


num_labels_task = {
    "sst-2": 2,
    "imdb": 2,
    "cola":2, 
    "agnews": 4,
    "yahoo": 10,
    "yelp": 5
}

    
 
