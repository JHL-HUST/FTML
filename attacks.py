import numpy as np
import copy

class Adversary(object):
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        self.synonym_selector = synonym_selector
        self.target_model = target_model
        self.max_perturbed_percent = max_perturbed_percent

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.
        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x

    def check_diff(self, sentence, perturbed_sentence):
        words = sentence.split()
        perturbed_words = perturbed_sentence.split()
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count

class GAAdversary(Adversary):
    """  GA attack method.  """

    def __init__(self, synonym_selector, target_model, iterations_num=20, pop_max_size=60, max_perturbed_percent=0.25):
        super(GAAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.temp = 0.3

    def predict_batch(self, sentences): # Done
        seqs = [" ".join(words) for words in sentences]
        tem, _ = self.target_model.query(seqs, None)
        tem = self._softmax(tem)
        return tem

    def predict(self, sentence): # Done
        tem, _ = self.target_model.query([" ".join(sentence)], None)
        tem = self._softmax(tem[0])
        return tem

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        new_x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]
        new_x_scores = new_x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur
    
    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def check_return(self, perturbed_words, ori_words, ori_label):
        perturbed_text = " ".join(perturbed_words)
        clean_text = " ".join(ori_words)
        if self.check_diff(clean_text, perturbed_text) / len(ori_words) > self.max_perturbed_percent:
            return False, clean_text, ori_label
        else:
            adv_label = self.target_model.query([perturbed_text], [ori_label])[1][0]
            assert (adv_label != ori_label)
            return True, perturbed_text, adv_label

    def run(self, sentence, ori_label):

        # x_orig = np.array(sentence.split())
        x_orig = sentence.split()
        x_len = len(x_orig)

        neigbhours_list = []
        for i in range(x_len):
            neigbhours_list.append(self.synonym_selector.find_synonyms(x_orig[i]))
            
        neighbours_len = [len(x) for x in neigbhours_list]
        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))

        if np.sum(w_select_probs) == 0:
            return False, sentence, ori_label
            
        w_select_probs = w_select_probs / np.sum(w_select_probs)

        pop = self.generate_population(
            x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        for i in range(self.max_iters):
            pop_preds = self.predict_batch(pop)
            pop_scores = 1 - pop_preds[:, ori_label]
            print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) != ori_label:
                return self.check_return(pop[top_attack], x_orig, ori_label)
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            childs = [self.perturb(
                x, x_orig, neigbhours_list, w_select_probs, ori_label) for x in childs]
            pop = elite + childs

        return False, sentence, ori_label


class PSOAdversary(Adversary):
    """ Particle Swarm Optimization Attack. """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25, pop_max_size=60, iterations_num=40, max_seq_length=500):
        
        super(PSOAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.max_seq_length = max_seq_length

    def do_replace(self, x_cur, pos, new_word): # Done
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def predict_batch(self, sentences): # Done
        seqs = [" ".join(words) for words in sentences]
        tem, _ = self.target_model.query(seqs, None)
        tem = self._softmax(tem)
        return tem

    def predict(self, sentence): # Done
        tem, _ = self.target_model.query([" ".join(sentence)], None)
        tem = self._softmax(tem[0])
        return tem

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """

        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]

        new_x_scores = x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            best_id = np.argsort(new_x_scores)[-1]
            if np.argmax(new_x_preds[best_id]) != ori_label:
                return [1, new_x_list[best_id]]
            return [x_scores[best_id], new_x_list[best_id]]
        return [orig_score, x_cur]

    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        
        x_len = w_select_probs.shape[0]
 
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        pop = []
        pop_scores=[]
        for i in range(pop_size):
            tem = self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label)
            if tem is None:
                return None
            if tem[0] == 1:
                return [tem[1]]
            else:
                pop_scores.append(tem[0])
                pop.append(tem[1])
        return pop_scores, pop

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(np.array(x) != np.array(x_orig))) / float(x_len)
        return change_ratio

    def check_return(self, perturbed_words, ori_words, ori_label):
        perturbed_text = " ".join(perturbed_words)
        clean_text = " ".join(ori_words)
        if self.check_diff(clean_text, perturbed_text) / len(ori_words) > self.max_perturbed_percent:
            return False, clean_text, ori_label
        else:
            adv_label = self.target_model.query([perturbed_text], [ori_label])[1][0]
            assert (adv_label != ori_label)
            return True, perturbed_text, adv_label

    def run(self, sentence, ori_label):

        # x_orig = np.array(sentence.split())
        x_orig = sentence.split()
        x_len = len(x_orig)
        
        neigbhours_list = []
        for i in range(x_len):
            neigbhours_list.append(self.synonym_selector.find_synonyms(x_orig[i]))

        neighbours_len = [len(x) for x in neigbhours_list]

        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))

        if np.sum(w_select_probs) == 0:
            return False, sentence, ori_label

        w_select_probs = w_select_probs / np.sum(w_select_probs)

        if np.sum(neighbours_len) == 0:
            return False, sentence, ori_label

        # print(neighbours_len)

        tem = self.generate_population(x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        if tem is None:
            return False, sentence, ori_label
        if len(tem) == 1:
            return self.check_return(tem[0], x_orig, ori_label)
        pop_scores, pop = tem
        part_elites = copy.deepcopy(pop)
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[-1]
        all_elite = pop[top_attack]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):

            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)

            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            # pop_scores = []
            # pop_scores_all=[]
            # for a in pop:
            #     pt = self.predict(a)
            #     pop_scores.append(1 - pt[ori_label])
            #     pop_scores_all.append(pt)

            pop_scores_all = self.predict_batch(pop)
            pop_scores = 1 - pop_scores_all[:, ori_label]

            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]

            # print('\t\t', i, ' -- ', pop_scores[top_attack])
            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                if np.argmax(pt) != ori_label:
                    return self.check_return(pop[pt_id], x_orig, ori_label)

            new_pop = []
            new_pop_scores=[]
            for id in range(len(pop)):
                x=pop[id]
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    tem = self.perturb(x, x_orig, neigbhours_list, w_select_probs, ori_label)
                    if tem is None:
                        return False, sentence, ori_label
                    if tem[0] == 1:
                        return self.check_return(tem[1], x_orig, ori_label)
                    else:
                        new_pop_scores.append(tem[0])
                        new_pop.append(tem[1])
                else:
                    new_pop_scores.append(pop_scores[id])
                    new_pop.append(x)
            pop = new_pop

            pop_scores = new_pop_scores
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)
        
        return False, sentence, ori_label

class PWWSAdversary(Adversary):
    """  PWWS attack method.  """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        super(PWWSAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)

    def R_func(self, clean_tokens, idx, candidates, ori_label):
        max_diff = -100
        max_word = clean_tokens[idx]
        sentence = ' '.join(clean_tokens)
        logits = self.target_model.query([sentence], [str(ori_label)])[0]
        score = self._softmax(logits)[0][ori_label]
        sentence_new_list = []
        for c in candidates:
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[idx] = c
            sentence_new = ' '.join(clean_tokens_new)
            sentence_new_list.append(sentence_new)
        if len(sentence_new_list) != 0:
            logits_new = self.target_model.query(sentence_new_list, [str(ori_label)]*len(sentence_new_list))[0]
            score_new = self._softmax(logits_new)[:, ori_label]
            diff = score - score_new
            max_diff = np.max(diff)
            max_word = candidates[np.argmax(diff)]
        return max_word, max_diff


    def S_func(self, clean_tokens, ori_label):
        saliency_list = []
        sentence = ' '.join(clean_tokens)
        logits = self.target_model.query([sentence], [str(ori_label)])[0]
        score = self._softmax(logits)[0][ori_label]
        sentence_new_list = []
        for i in range(len(clean_tokens)):
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[i] = '[UNK]'
            sentence_new = ' '.join(clean_tokens_new)
            sentence_new_list.append(sentence_new)
        logits_new = self.target_model.query(sentence_new_list, [str(ori_label)]*len(sentence_new_list))[0]
        score_new = self._softmax(logits_new)[:, ori_label]
        saliency = score - score_new
        soft_saliency_list = list(self._softmax(saliency))
        return soft_saliency_list
    

    def H_func(self, clean_tokens, ori_label):
        saliency_list = self.S_func(clean_tokens, ori_label)
        result_list = []
        for i, w in enumerate(clean_tokens):
            candidates = self.synonym_selector.find_synonyms(w)
            max_word, max_diff = self.R_func(clean_tokens, i, candidates, ori_label)
            result_list.append([i, max_word, max_diff])
        score_list = [res[2] * saliency for res, saliency in zip(result_list, saliency_list)]
        indexes = np.argsort(np.array(score_list))[::-1]
        replace_list = []
        for index in indexes:
            res = result_list[index]
            replace_list.append([res[0], res[1]])
        return replace_list

    def run(self, sentence, ori_label): 
        clean_tokens = sentence.split()
        adv_sentence = sentence
        adv_label = ori_label
        success = False
        perturbed_tokens = list(clean_tokens)

        replace_list = self.H_func(clean_tokens, ori_label)
        for i in range(len(replace_list)):
            tmp = replace_list[i]
            perturbed_tokens[tmp[0]] = tmp[1]
            adv_sentence = ' '.join(perturbed_tokens)
            prediction = self.target_model.query([adv_sentence], [str(ori_label)])[1][0]
            if int(prediction) != int(ori_label):
                success = True
                adv_label = prediction
                return success, adv_sentence, adv_label
            if self.check_diff(sentence, adv_sentence) + 1 > len(clean_tokens) * self.max_perturbed_percent:
                break
        return success, adv_sentence, adv_label

class HLAAdversary(Adversary):
    """ Hard Label Attack. """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        super(HLAAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)

        self.pos_filter = False
        self.USE_cache_path = './data/aux_files/cache_USE'
        self.use = USEmodel(self.USE_cache_path)

        if self.pos_filter:
            self.idx2word = {}
            self.word2idx = {}
            # self.counter_fitting_embeddings_path = "/home/yuzhen/nlp/synonym/counter-fitted-vectors.txt"
            # self.counter_fitting_cos_sim_path = "/home/yuzhen/nlp/synonym/counter-fitting-cos-sim.txt"
            self.counter_fitting_embeddings_path = None
            self.counter_fitting_cos_sim_path = None
            print("Building vocab...")
            with open(self.counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    word = line.split()[0]
                    if word not in self.idx2word:
                        self.idx2word[len(self.idx2word)] = word
                        self.word2idx[word] = len(self.idx2word) - 1
            print('Load pre-computed cosine similarity matrix from {}'.format(self.counter_fitting_cos_sim_path))
            with open(self.counter_fitting_cos_sim_path, "rb") as fp:
                self.cos_sim = pickle.load(fp)
        else:
            self.idx2word = synonym_selector.inv_vocab
            self.word2idx = synonym_selector.vocab
            self.cos_sim = None


    def run(self, sentence, ori_label):
        x_orig = sentence.split()
        x_len = len(x_orig)

        new_text, num_changed, random_changed, _, \
        new_label, num_queries, sim, random_sim = self.attack(x_orig, ori_label,self.word2idx, self.idx2word, self.cos_sim, sim_predictor=self.use)

        if new_label != ori_label and self.check_diff(sentence, new_text) / x_len <= self.max_perturbed_percent:
            return True, new_text, new_label
        else:
            return False, sentence, ori_label


    def attack(self, text_ls, true_label, word2idx, idx2word, cos_sim, sim_predictor=None, sim_score_window=40, synonym_num=4, top_k_words = 1000000):

        assert(isinstance(text_ls, list))

        # first check the prediction of the original text
        orig_probs, ori_pred = self.target_model.query([" ".join(text_ls)], None)
        orig_label = ori_pred[0]
        orig_prob = orig_probs.max()
        if true_label != orig_label:
            return " ".join(text_ls), 0, 0, orig_label, orig_label, 0, 0, 0
        else:
            len_text = len(text_ls)
            if len_text < sim_score_window:
                sim_score_threshold = 0.1  # shut down the similarity thresholding function
            half_sim_score_window = (sim_score_window - 1) // 2
            num_queries = 1
            rank = {}
            # get the pos and verb tense info
            if self.pos_filter:
                words_perturb = []
                pos_ls = get_pos(text_ls)
                pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
                for pos in pos_pref:
                    for i in range(len(pos_ls)):
                        if pos_ls[i] == pos and len(text_ls[i]) > 2:
                            words_perturb.append((i, text_ls[i]))
            else:
                words_perturb = []
                for i, word in enumerate(text_ls):
                    syns = self.synonym_selector.find_synonyms(word)
                    if len(syns) > 0:
                        words_perturb.append((i, word))
                pos_ls = None

            random.shuffle(words_perturb)
            # find synonyms and make a dict of synonyms of each word.
            words_perturb = words_perturb[:top_k_words]
            # words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
            # words_perturb_idx = [word2idx[word] if word in word2idx else 0 for idx, word in words_perturb]
            # synonym_words, synonym_values = [], []
            # for idx in words_perturb_idx:
            #     res = list(zip(*(cos_sim[idx])))
            #     temp=[]
            #     for ii in res[1]:
            #         temp.append(idx2word[ii])
            #     synonym_words.append(temp[:synonym_num+1])
            #     temp=[]
            #     for ii in res[0]:
            #         temp.append(ii)
            #     synonym_values.append(temp[:synonym_num+1])
            # synonyms_all = []
            # synonyms_dict = defaultdict(list)
            # for idx, word in words_perturb:
            #     if word in word2idx:
            #         synonyms = synonym_words.pop(0)
            #         if synonyms:
            #             synonyms_all.append((idx, synonyms))
            #             synonyms_dict[word] = synonyms
            synonyms_all = []
            synonyms_dict = defaultdict(list)
            for idx, word in words_perturb:
                synonyms = self.synonym_selector.find_synonyms(word)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

            # STEP 1: Random initialisation.
            qrs = 0
            num_changed = 0
            flag = 0
            th = 0
            # Try substituting a random index with its random synonym.
            while qrs < len(text_ls):
                random_text = text_ls[:]
                for i in range(len(synonyms_all)):
                    idx = synonyms_all[i][0]
                    syn = synonyms_all[i][1]
                    random_text[idx] = random.choice(syn)
                    if i >= th:
                        break
                pr = self.get_attack_result([random_text], orig_label)
                qrs+=1
                th +=1
                if th > len_text:
                    break
                if np.sum(pr)>0:
                    flag = 1
                    break
            old_qrs = qrs
            # If adversarial text is not yet generated try to substitute more words than 30%.
            while qrs < old_qrs + 2500 and flag == 0:
                random_text = text_ls[:]
                for j in range(len(synonyms_all)):
                    idx = synonyms_all[j][0]
                    syn = synonyms_all[j][1]
                    random_text[idx] = random.choice(syn)
                    if j >= len_text:
                        break
                pr = self.get_attack_result([random_text], orig_label)
                qrs+=1
                if np.sum(pr)>0:
                    flag = 1
                    break

            if flag == 1:
                #print("Found "+str(sample_index))
                changed = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=random_text[i]:
                        changed+=1
                print("Step 1 Changed_num: "+ str(changed))

                # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
                while True:
                    choices = []

                    # For each word substituted in the original text, change it with its original word and compute
                    # the change in semantic similarity.
                    for i in range(len(text_ls)):
                        if random_text[i] != text_ls[i]:
                            new_text = random_text[:]
                            new_text[i] = text_ls[i]
                            semantic_sims = self.calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                            qrs+=1
                            pr = self.get_attack_result([new_text], orig_label)
                            if np.sum(pr) > 0:
                                choices.append((i,semantic_sims[0]))

                    # Sort the relacements by semantic similarity and replace back the words with their original
                    # counterparts till text remains adversarial.
                    if len(choices) > 0:
                        choices.sort(key = lambda x: x[1])
                        choices.reverse()
                        for i in range(len(choices)):
                            new_text = random_text[:]
                            new_text[choices[i][0]] = text_ls[choices[i][0]]
                            pr = self.get_attack_result([new_text], orig_label)
                            qrs+=1
                            if pr[0] == 0:
                                break
                            random_text[choices[i][0]] = text_ls[choices[i][0]]

                    if len(choices) == 0:
                        break

                changed_indices = []
                num_changed = 0
                for i in range(len(text_ls)):
                    if text_ls[i]!=random_text[i]:
                        changed_indices.append(i)
                        num_changed+=1
                print("Step2 change_num/len:"+str(num_changed)+"/"+str(len(text_ls))+"     query_time:"+str(qrs)) 
                # print(str(num_changed)+" "+str(qrs))
                random_sim = self.calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
                #return '', 0, orig_label, orig_label, 0
                if num_changed == 1:
                    _, out_ret_lable = self.target_model.query([" ".join(random_text)], None)
                    return ' '.join(random_text), 1, 1, \
                        orig_label, out_ret_lable[0], qrs, random_sim, random_sim
                population_size = 30
                population = []
                old_syns = {}
                max_replacements = defaultdict(int)
                # STEP 3: Genetic Optimization
                # Genertaes initial population by mutating the substituted indices.
                for i in range(len(changed_indices)):
                    txt, mut_qrs = self.mutate(changed_indices[i], text_ls, pos_ls, random_text, random_text, changed_indices,
                                    synonyms_dict, old_syns, orig_label, sim_score_window, sim_predictor)
                    qrs+=mut_qrs
                    if len(txt)!=0:
                        population.append(txt)
                max_iters = 100
                pop_count = 0
                attack_same = 0
                old_best_attack = random_text[:]
                if len(population) == 0:
                    _,out_ret_lable =self.target_model.query([" ".join(random_text)], None)
                    return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                                orig_label, out_ret_lable[0], qrs, random_sim, random_sim

                ## Genetic Optimization
                for _ in range(max_iters):
                    max_changes = len_text

                    # Find the best_attack text in the current population.
                    for txt in population:
                        changes = 0
                        for i in range(len(changed_indices)):
                            j = changed_indices[i]
                            if txt[j] != text_ls[j]:
                                changes+=1
                        if changes <= max_changes:
                            max_changes = changes
                            best_attack = txt

                    pr = self.get_attack_result([best_attack], orig_label)

                    # Check that it is adversarial.
                    assert pr[0] > 0
                    flag = 0

                    # If the new best attack is the same as the old best attack for last 15 consecutive iterations tham
                    # stop optimization.
                    for i in range(len(changed_indices)):
                        k = changed_indices[i]
                        if best_attack[k] != old_best_attack[k]:
                            flag = 1
                            break
                    if flag == 1:
                        attack_same = 0
                    else:
                        attack_same+=1

                    if attack_same >= 15:
                        sim = self.calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                        num_changed = 0
                        for i in range(len(text_ls)): 
                            if text_ls[i] != best_attack[i]:                                                                                              
                                num_changed += 1 
                        print("Step3 change_num/len:"+str(num_changed)+"/"+str(len(text_ls)))
                        _, out_ret_lable = self.target_model.query([" ".join(best_attack)], None)
                        return ' '.join(best_attack), max_changes, len(changed_indices), \
                            orig_label, out_ret_lable[0], qrs, sim, random_sim

                    old_best_attack = best_attack[:]

                    #print(str(max_changes)+" After Genetic")

                    # If only 1 input word substituted return it.
                    if max_changes == 1:
                        sim = self.calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                        num_changed = 0
                        for i in range(len(text_ls)): 
                            if text_ls[i] != best_attack[i]:                                                                                              
                                num_changed += 1 
                        print("Step3 change_num/len:"+str(num_changed)+"/"+str(len(text_ls)))
                        return ' '.join(best_attack), max_changes, len(changed_indices), \
                            orig_label, self.target_model.query([" ".join(best_attack)], None)[1][0], qrs, sim, random_sim


                    # Sample two parent input propotional to semantic similarity.
                    sem_scores = self.calc_sim(text_ls, population, -1, sim_score_window, sim_predictor)
                    sem_scores = np.asarray(sem_scores)
                    scrs = self._softmax(sem_scores)
                    parent1_idx = np.random.choice(len(population), size = population_size-1, p = scrs)
                    parent2_idx = np.random.choice(len(population), size = population_size-1, p = scrs)

                    ## Crossover
                    final_childs, cross_qrs = self.crossover(population_size, population, parent1_idx, parent2_idx,
                                            text_ls, best_attack, max_changes, changed_indices, sim_score_window, sim_predictor, orig_label)
                    qrs+=cross_qrs
                    population = []
                    indices_done = []

                    # Randomly select indices for mutation from the changed indices. The changed indices contains indices
                    # which has not been replaced by original word.
                    indices = np.random.choice(len(changed_indices), size = min(len(changed_indices), len(final_childs)))
                    for i in range(len(indices)):
                        child = final_childs[i]
                        j = indices[i]
                        # If the index has been substituted no need to mutate.
                        if text_ls[changed_indices[j]] == child[changed_indices[j]]:
                            population.append(child)
                            indices_done.append(j)
                            continue
                        txt = []
                        # Mutate the childs obtained after crossover on the random index.
                        if max_replacements[changed_indices[j]] <= 25:
                            txt, mut_qrs = self.mutate(changed_indices[j], text_ls, pos_ls, child, child, changed_indices,
                                                synonyms_dict, old_syns, orig_label, sim_score_window, sim_predictor)
                        qrs+=mut_qrs
                        indices_done.append(j)

                        # If the input has been mutated successfully add to population for nest generation.
                        if len(txt)!=0:
                            max_replacements[changed_indices[j]] +=1
                            population.append(txt)
                    if len(population) == 0:
                        pop_count+=1
                    else:
                        pop_count = 0

                    # If length of population is zero for 15 consecutive iterations return.
                    if pop_count >= 15:
                        sim = self.calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                        num_changed = 0
                        for i in range(len(text_ls)): 
                            if text_ls[i]!=best_attack[i]:                                                                                              
                                num_changed+=1 
                        print("Step3 change_num/len:"+str(num_changed)+"/"+str(len(text_ls)))
                        _,out_ret_lable =self.target_model.query([" ".join(best_attack)], None)
                        return ' '.join(best_attack), len(changed_indices), \
                            max_changes, orig_label, out_ret_lable[0], qrs, sim, random_sim

                    # Add best adversarial attack text also to next population.
                    population.append(best_attack)
                sim = self.calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

                num_changed = 0
                for i in range(len(text_ls)): 
                    if text_ls[i]!=best_attack[i]:                                                                                              
                        num_changed+=1 
                print("Step3 change_num/len:"+str(num_changed)+"/"+str(len(text_ls)))
                _,out_ret_lable =self.target_model.query([" ".join(best_attack)], None)
                return ' '.join(best_attack), max_changes, len(changed_indices), \
                    orig_label, out_ret_lable[0], qrs, sim, random_sim

            else:
                print("Not Found")
                return " ".join(text_ls), 0,0, orig_label, orig_label, 0, 0, 0


    def calc_sim(self, text_ls, new_texts, idx, sim_score_window, sim_predictor):

        len_text = len(text_ls)
        half_sim_score_window = (sim_score_window - 1) // 2

        # Compute the starting and ending indices of the window.
        if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = idx - half_sim_score_window
            text_range_max = idx + half_sim_score_window + 1
        elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = 0
            text_range_max = sim_score_window
        elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
            text_range_min = len_text - sim_score_window
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text

        if text_range_min < 0:
            text_range_min = 0
        if text_range_max > len_text:
            text_range_max = len_text

        if idx == -1:
            text_rang_min = 0
            text_range_max = len_text
        # Calculate semantic similarity using USE.
        semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

        return semantic_sims

    def get_attack_result(self, new_text, orig_label):
        new_text = [" ".join(text) for text in new_text]
        tem, templable = self.target_model.query(new_text, None)
        pr = np.array([1 if orig_label!= templable[i] else 0 for i in range(len(templable))])
        # pr2=(orig_label!= templable)
        # assert(pr == pr2)
        return pr

    def mutate(self, rand_idx, text_ls, pos_ls, new_attack, best_attack, remaining_indices,
            synonyms_dict, old_syns, orig_label, sim_score_window, sim_predictor):

        # Calculates the semantic similarity before mutation.
        random_text = new_attack[:]
        syns = synonyms_dict[text_ls[rand_idx]]
        prev_semantic_sims = self.calc_sim(text_ls, [best_attack], rand_idx, sim_score_window, sim_predictor)
        # Gives Priority to Original Word
        orig_word = 0
        if random_text[rand_idx] != text_ls[rand_idx]:

            temp_text = random_text[:]
            temp_text[rand_idx] = text_ls[rand_idx]
            pr = self.get_attack_result([temp_text], orig_label)
            semantic_sims = self.calc_sim(text_ls, [temp_text], rand_idx, sim_score_window, sim_predictor)
            if np.sum(pr) > 0:
                orig_word = 1
                return temp_text, 1  #(updated_text, queries_taken)

        # If replacing with original word does not yield adversarial text, then try to replace with other synonyms.
        if orig_word == 0:
            final_mask = []
            new_texts = []
            final_texts = []

            # Replace with synonyms.
            for syn in syns:

                # Ignore the synonym already present at position rand_idx.
                if syn == best_attack[rand_idx]:
                    final_mask.append(0)
                else:
                    final_mask.append(1)
                temp_text = random_text[:]
                temp_text[rand_idx] = syn
                new_texts.append(temp_text[:])

            # Filter out mutated texts that: (1) are not having same POS tag of the synonym, (2) lowers Semantic Similarity and (3) Do not satisfy adversarial criteria.

            semantic_sims = self.calc_sim(text_ls, new_texts, rand_idx, sim_score_window, sim_predictor)
            pr = self.get_attack_result(new_texts, orig_label)
            final_mask = np.asarray(final_mask)
            sem_filter = semantic_sims >= prev_semantic_sims[0]
            prediction_filter = pr > 0
            final_mask = final_mask*sem_filter
            final_mask = final_mask*prediction_filter
            if self.pos_filter:
                synonyms_pos_ls = [get_pos(new_text[max(rand_idx - 4, 0):rand_idx + 5])[min(4, rand_idx)]
                                    if len(new_text) > 10 else get_pos(new_text)[rand_idx] for new_text in new_texts]
                pos_mask = np.array(pos_filter(pos_ls[rand_idx], synonyms_pos_ls))
                final_mask = final_mask*pos_mask
            sem_vals = final_mask*semantic_sims

            for i in range(len(sem_vals)):
                if sem_vals[i] > 0:
                    final_texts.append((new_texts[i], sem_vals[i]))

            # Return mutated text with best semantic similarity.
            final_texts.sort(key =  lambda x : x[1])
            final_texts.reverse()

            if len(final_texts) > 0:
                #old_syns[rand_idx].append(final_texts[0][0][rand_idx])
                return final_texts[0][0], len(new_texts)
            else:
                return [], len(new_texts)

    def crossover(self, population_size, population, parent1_idx, parent2_idx,
                text_ls, best_attack, max_changes, changed_indices,
                sim_score_window, sim_predictor, orig_label):

        childs = []
        changes = []

        # Do crossover till population_size-1.
        for i in range(population_size-1):

            # Generates new child.
            p1 = population[parent1_idx[i]]
            p2 = population[parent2_idx[i]]
            assert len(p1) == len(p2)
            new_child = []
            for j in range(len(p1)):
                if np.random.uniform() < 0.5:
                    new_child.append(p1[j])
                else:
                    new_child.append(p2[j])
            change = 0
            cnt = 0
            mismatches = 0
            # Filter out crossover child which (1) Do not improve semantic similarity, (2) Have number of words substituted
            # more than the current best_attack.
            for k in range(len(changed_indices)):
                j = changed_indices[k]
                if new_child[j] == text_ls[j]:
                    change+=1
                    cnt+=1
                elif new_child[j] == best_attack[j]:
                    change+=1
                    cnt+=1
                elif new_child[j] != best_attack[j]:
                    change+=1
                    prev_semantic_sims = self.calc_sim(text_ls, [best_attack], j, sim_score_window, sim_predictor)
                    semantic_sims = self.calc_sim(text_ls, [new_child], j, sim_score_window, sim_predictor)
                    if semantic_sims[0] >= prev_semantic_sims[0]:
                        mismatches+=1
                        cnt+=1
            if cnt==change and mismatches<=max_changes:
                childs.append(new_child)
            changes.append(change)
        if len(childs) == 0:
            return [], 0

        # Filter out childs whoch do not satisfy the adversarial criteria.
        pr = self.get_attack_result(childs, orig_label)
        final_childs = [childs[i] for i in range(len(pr)) if pr[i] > 0]
        return final_childs, len(childs)