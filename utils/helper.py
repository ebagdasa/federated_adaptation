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
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'

        # TRAINING PARAMS
        self.lr = self.params.get('lr', None)
        self.decay = self.params.get('decay', None)
        self.momentum = self.params.get('momentum', None)
        self.epochs = self.params.get('epochs', None)
        self.is_save = self.params.get('save_model', False)
        self.log_interval = self.params.get('log_interval', 1000)
        self.batch_size = self.params.get('batch_size', None)
        self.test_batch_size = self.params.get('test_batch_size', None)
        self.optimizer = self.params.get('optimizer', None)
        self.scheduler = self.params.get('scheduler', False)
        self.resumed_model = self.params.get('resumed_model', False)

        # LOGGING
        self.log = self.params.get('log', True)
        self.tb = self.params.get('tb', True)
        self.random = self.params.get('random', True)
        self.report_train_loss = self.params.get('report_train_loss', True)


        self.data_type = self.params.get('data_type', 'image')
        self.start_epoch = 1

        ### FEDERATED LEARNING PARAMS
        self.sampling_dirichlet = self.params.get('sampling_dirichlet', False)
        self.number_of_total_participants = self.params.get('number_of_total_participants', None)
        self.no_models = self.params.get('no_models', None)
        self.retrain_no_times = self.params.get('retrain_no_times', 1)
        self.eta = self.params.get('eta', 1)
        self.diff_privacy = self.params.get('diff_privacy', False)

        ### TEXT PARAMS
        self.bptt = self.params.get('bptt', False)
        self.recreate_dataset = self.params.get('recreate_dataset', False)


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

    def save_model(self, epoch=0, val_loss=0):
        model = self.target_model
        if self.is_save and self.log:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.get('save_on_epochs', []):
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
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

    # def compute_rdp(self):
    #     from compute_dp_sgd_privacy import apply_dp_sgd_analysis
    #
    #     N = self.dataset_size
    #     logger.info(f'Dataset size: {N}. Computing RDP guarantees.')
    #     q = self.params['batch_size'] / N  # q - the sampling ratio.
    #
    #     orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
    #               list(range(5, 64)) + [128, 256, 512])
    #
    #     steps = int(math.ceil(self.params['epochs'] * N / self.params['batch_size']))
    #
    #     apply_dp_sgd_analysis(q, self.params['z'], steps, orders, 1e-6)

    @staticmethod
    def clip_grad(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm

    @staticmethod
    def clip_grad_scale_by_layer_norm(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm_weight = 0
        norm_weight = dict()
        for i, p in enumerate(parameters):
            param_norm = p.data.norm(norm_type)
            norm_weight[i] = param_norm.item()
            total_norm_weight += param_norm.item() ** norm_type
        total_norm_weight = total_norm_weight ** (1. / norm_type)

        total_norm = 0
        norm_grad = dict()
        for i, p in enumerate(parameters):
            param_norm = p.grad.data.norm(norm_type)
            norm_grad[i] = param_norm.item()
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for i, p in enumerate(parameters):
                if norm_grad[i] < 1e-3:
                    continue
                scale = norm_weight[i] / total_norm_weight
                p.grad.data.mul_(math.sqrt(max_norm) * scale / norm_grad[i])
        # print(total_norm)
        # total_norm = 0
        # norm_grad = dict()
        # for i, p in enumerate(parameters):
        #     param_norm = p.grad.data.norm (norm_type)
        #     norm_grad[i] = param_norm
        #     total_norm += param_norm.item() ** norm_type
        # total_norm = total_norm ** (1. / norm_type)
        # print(total_norm)
        return total_norm



    def get_grad_vec(self, model):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if self.device.type == 'cpu':
            sum_var = torch.FloatTensor(size).fill_(0)
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    def get_optimizer(self, model):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.lr,
                                  weight_decay=self.decay, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def check_resume_training(self, model, lr=False):
        from models.resnet import ResNet18

        if self.resumed_model:
            logger.info('Resuming training...')
            loaded_params = torch.load(f"saved_models/{self.resumed_model}")
            model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            if lr:
                self.lr = loaded_params.get('lr', self.lr)

            self.fixed_model = ResNet18()
            self.fixed_model.to(self.device)
            self.fixed_model.load_state_dict(loaded_params['state_dict'])

            logger.warning(f"Loaded parameters from saved model: LR is"
                        f" {self.lr} and current epoch is {self.start_epoch}")

    def flush_writer(self):
        if self.writer:
            self.writer.flush()

    def plot(self, x, y, name):
        if self.writer is not None:
            self.writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    @staticmethod
    def fix_random(seed=0):
        # logger.warning('Setting random seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        return True

    def average_shrink_models(self, weight_accumulator, target_model, epoch):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * \
                               (self.eta / self.number_of_total_participants)

            if self.diff_privacy:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            data.add_(update_per_layer)

        return True


    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            if 'running_' in name or '_tracked' in name:
                continue
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum


    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)


    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = target_vec.clone().detach()
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(self.params['scale_weights']*(model_vec-target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info( cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1-cs_sim

        return 1e3*loss



    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100*(data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[name].view(-1)


            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list)/len(cs_list)).data[0])
        return 1e3*sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():

            cs = cs_loss(last_acc[name].view(-1),
                         new_acc[name].view(-1)
                         )
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)

