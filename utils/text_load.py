import os
import torch
import json
import re
from tqdm import tqdm
import random

filter_symbols = re.compile('[a-zA-Z]*')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)


def get_word_list(line, dictionary):
    splitted_words = json.loads(line.lower()).split()
    words = ['<bos>']
    for word in splitted_words:
        word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words


class Corpus(object):
    def __init__(self, params, dictionary):       
        repopath = params['repo_path']
        self.path = f'{repopath}/data'
        authors_no = params['number_of_total_participants']
        self.local_test_perc = params['local_test_perc']
        self.dictionary = dictionary
        self.no_tokens = len(self.dictionary)
        self.authors_no = authors_no
        self.auxiliary = self.tokenize_aux(os.path.join(self.path, 'test_data.json'))
        self.train, self.test, self.diff_words, self.voc_size = self.tokenize_train(f'{self.path}/shard_by_author')

    def tokenize_train(self, path):
        """
        We return a list of ids per each participant.
        :param path:
        :return:
        """
        files = os.listdir(path)
        per_participant_ids = list()
        per_participant_ids_test = list()
        per_participant_different_words = list()
        per_participant_voc_size = list()
        for file in tqdm(files[:self.authors_no]):
            # jupyter creates somehow checkpoints in this folder
            if 'checkpoint' in file:
                continue
            new_path=f'{path}/{file}'
            with open(new_path, 'r') as f:
                diff_word = 0
                tokens = 0
                word_list = list()
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    tokens += len(words)
                    wordidx = [self.dictionary.word2idx[x] for x in words]
                    diff_word += sum([i not in word_list for i in wordidx])
                    word_list.extend(wordidx)
                ids = torch.LongTensor(word_list)
                ids_test = torch.LongTensor(word_list[len(word_list)//100*(100-self.local_test_perc):])
            if len(ids)>=10:
                per_participant_ids.append(ids)
                per_participant_ids_test.append(ids_test)
                per_participant_different_words.append(diff_word)
                per_participant_voc_size.append(tokens)
        return per_participant_ids, per_participant_ids_test, per_participant_different_words, per_participant_voc_size

    def tokenize_aux(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        word_list = list()
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = get_word_list(line, self.dictionary)
                tokens += len(words)
                word_list.extend([self.dictionary.word2idx[x] for x in words])
        ids = torch.LongTensor(word_list)
        return ids