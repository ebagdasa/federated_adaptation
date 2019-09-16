import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from utils.helper import Helper
import random
import logging

from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
from utils.text_load import *

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


class TextHelper(Helper):
    corpus = None

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()


    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])

        # logger.info(' '.join(result))
        return ' '.join(result)

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.params['bptt'], len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target


    def load_data(self):
        ### DATA PART

        logger.info('Loading data')
        #### check the consistency of # of batches and size of dataset for poisoning

        dictionary = torch.load(self.params['word_dictionary_path'])
        corpus_file_name = f"{self.params['data_folder']}/" \
            f"corpus_{self.params['number_of_total_participants']}.pt.tar"
        if self.recreate_dataset:

            self.corpus = Corpus(self.params, dictionary=dictionary,
                                 is_poison=False)
            torch.save(self.corpus, corpus_file_name)
        else:
            self.corpus = torch.load(corpus_file_name)
        logger.info('Loading data. Completed.')
        ### PARSE DATA
        eval_batch_size = self.test_batch_size
        self.train_data = [self.batchify(data_chunk, self.batch_size) for data_chunk in
                           self.corpus.train]
        self.test_data = self.batchify(self.corpus.test, eval_batch_size)

        self.n_tokens = len(self.corpus.dictionary)

    def create_model(self):

        local_model = RNNModel(name='Local_Model', created_time=self.params['current_time'],
                               rnn_type='LSTM', ntoken=self.n_tokens,
                               ninp=self.params['emsize'], nhid=self.params['nhid'],
                               nlayers=self.params['nlayers'],
                               dropout=self.params['dropout'], tie_weights=self.params['tied'])
        local_model.cuda()
        target_model = RNNModel(name='Target', created_time=self.params['current_time'],
                                rnn_type='LSTM', ntoken=self.n_tokens,
                                ninp=self.params['emsize'], nhid=self.params['nhid'],
                                nlayers=self.params['nlayers'],
                                dropout=self.params['dropout'], tie_weights=self.params['tied'])
        target_model.cuda()
        if self.resumed_model:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
