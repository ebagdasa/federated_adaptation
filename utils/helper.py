import logging

logger = logging.getLogger('logger')
from torch.nn.functional import log_softmax
from shutil import copyfile

import math
import torch
import random
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd



class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None
        self.dataset_size = 0
        self.train_dataset = None
        self.test_dataset = None
        self.poisoned_data = None
        self.test_data_poison = None
        self.writer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.repo_path = self.params.get('repo_path', os.getcwd())
        self.folder_path = f'{self.repo_path}/saved_models/model_{self.name}_{current_time}'
        savename = self.params.get('save_name','debug')
        self.save_name = f'{self.repo_path}/stats/{savename}'

        # TRAINING PARAMS
        self.lr = self.params.get('lr', None)
        self.decay = self.params.get('decay', None)
        self.momentum = self.params.get('momentum', None)
        self.total_rounds = self.params.get('total_rounds', 0)
        self.save_on_rounds = self.params.get('save_on_rounds', [])
        self.is_save = self.params.get('save_model', False)
        self.batch_size = self.params.get('batch_size', None)
        self.test_batch_size = self.params.get('test_batch_size', None)
        self.optimizer = self.params.get('optimizer', None)
        self.resumed_model = self.params.get('resumed_model', False)
        
        self.local_test_perc = self.params.get('local_test_perc', 10)
        self.only_eval = self.params.get('only_eval', False)
        self.scratch = self.params.get('scratch', False)
        self.ewc = self.params.get('ewc', False)
        self.lamb = self.params.get('lamb', 5000)
        self.resumed_fisher = self.params.get('resumed_fisher', False)
        self.kd = self.params.get('kd', False)
        self.alpha = self.params.get('alpha', 0.95)
        self.temperature = self.params.get('temperature', 6)
        
        # LOGGING
        self.log = self.params.get('log', True)
        self.tb = self.params.get('tb', True)
        self.random = self.params.get('random', True)
        
        self.freeze_base = self.params.get('freeze_base', False)
        self.partial_test = self.params.get('partial_test', False)
        self.multi_gpu = self.params.get('multi_gpu', False)
        self.data_type = self.params.get('data_type', 'image')
        self.start_round = 1

        ### FEDERATED LEARNING PARAMS
        self.sampling_dirichlet = self.params.get('sampling_dirichlet', False)
        self.number_of_total_participants = self.params.get('number_of_total_participants', None)
        self.no_models = self.params.get('no_models', None)
        self.retrain_no_times = self.params.get('retrain_no_times', 1)
        self.adaptation_epoch = self.params.get('adaptation_epoch', 100)
        self.eta = self.params.get('eta', 1)

        ## Differential privacy
        self.diff_privacy = self.params.get('diff_privacy', False)
        self.s_norm = self.params.get('s_norm', 1)
        self.sigma = self.params.get('sigma', 1)

        ### TEXT PARAMS
        self.bptt = self.params.get('bptt', False)
        self.recreate_dataset = self.params.get('recreate_dataset', False)
        self.aggregation_type = self.params.get('aggregation_type', 'averaging')
        self.tied = self.params.get('tied', False)

        if self.log:
            try:
                os.mkdir(self.folder_path)
            except FileExistsError:
                logger.info('Folder already exists')
        else:
            self.folder_path = None

        # if not self.params.get('environment_name', False):
        #     self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_model(self, round=0, val_loss=0):
        model = self.target_model
        if self.is_save and self.log:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'round': round,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if round in self.save_on_rounds:
                logger.info(f'Saving model on round {round}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.round_{round}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.is_save:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def norm(parameters, max_norm):
        total_norm = 0
        for p in parameters:
            torch.sum(torch.pow(p))
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    @staticmethod
    def fix_random(seed=0):
        # logger.warning('Setting random seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        return True

    def average_shrink_models(self, weight_accumulator, target_model):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue
            if data.dtype != torch.float32:
                continue
                
            update_per_layer = weight_accumulator[name] * \
                               (self.eta / self.no_models)

            if self.diff_privacy:
                update_per_layer.add_(self.dp_noise(data, self.sigma))

            data.add_(update_per_layer)

        return True

    def median_aggregation(self, weight_accumulator, target_model):
        """
        Coordinate-wise median
        :param weight_accumulator:
        :param target_model:
        :return:
        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue
            if data.dtype != torch.float32:
                continue

            update_per_layer = self.eta * weight_accumulator[name].median(dim=0).values

            data.add_(update_per_layer.cuda())

        return True


    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            if 'running_' in name or '_tracked' in name:
                continue
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)