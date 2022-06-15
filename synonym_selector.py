import numpy as np

class SynonymSelector(object):
    """An class tries to find synonyms for a given word."""

    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']


    def find_synonyms(self, word):
        """Return the `num` nearest synonyms of word."""
        raise NotImplementedError


class EmbeddingSynonym(SynonymSelector):
    """Selecting syonyms by GLove word embeddings distance."""

    def __init__(self, max_candidates, vocab, inv_vocab, synonym_matrix, threshold=None):
        super(EmbeddingSynonym, self).__init__(vocab, inv_vocab)
        self.max_candidates = max_candidates
        self.synonym_matrix = synonym_matrix
        self.threshold = threshold

    def find_synonyms(self, word, syn_num=None, return_id=False):
        if word in self.stop_words or word not in self.vocab or word == 'UNK':
            return []
        word_id = self.vocab[word]
        dist_order = self.synonym_matrix[word_id, :, 0]
        dist_list = self.synonym_matrix[word_id, :, 1]
        if syn_num:
            n_return = np.min([np.sum(dist_order > 0), syn_num])
        else:
            n_return = np.min([np.sum(dist_order > 0), self.max_candidates])
        dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]
        if self.threshold is not None:
            mask_thres = np.where(dist_list < self.threshold)
            dist_order, dist_list = dist_order[mask_thres], dist_list[mask_thres]
        if return_id:
            return dist_order
        synonyms = []
        for word_id in dist_order:
            synonyms.append(self.inv_vocab[word_id])
        return synonyms

    def find_synonyms_for_tokens(self, words):
        synsets = []
        for w in words:
            synsets.append(self.find_synonyms(w))
        return synsets

    def find_synonyms_id(self, word_id, syn_num=None):
        if word_id <= 0 or word_id > 50000:
            return []
        dist_order = self.synonym_matrix[word_id, :, 0]
        dist_list = self.synonym_matrix[word_id, :, 1]
        if syn_num:
            n_return = np.min([np.sum(dist_order > 0), syn_num])
        else:
            n_return = np.min([np.sum(dist_order > 0), self.max_candidates])
        dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]
        if self.threshold is not None:
            mask_thres = np.where(dist_list < self.threshold)
            dist_order, dist_list = dist_order[mask_thres], dist_list[mask_thres]
        return np.array(dist_order).tolist()

        