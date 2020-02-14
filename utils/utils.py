import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import itertools
import matplotlib
matplotlib.use('AGG')
import logging
import colorlog
import os

def create_table(params: dict):
    header = f"| {' | '.join([x[:12] for x in params.keys() if x != 'folder_path'])} |"
    line = f"|{'|:'.join([3*'-' for x in range(len(params.keys())-1)])}|"
    values = f"| {' | '.join([str(params[x]) for x in params.keys() if x != 'folder_path'])} |"
    return '\n'.join([header, line, values])

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)

def fisher_matrix_diag(helper, data_source, target_model, criterion):
    model = target_model
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    start_time = time.time()
    if helper.data_type == 'text':
        if helper.multi_gpu:
            hidden = model.module.init_hidden(helper.params['test_batch_size'])
        else:
            hidden = model.init_hidden(helper.params['test_batch_size'])
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        # Forward and backward
        model.zero_grad()
        if helper.data_type == 'text':
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            loss = criterion(output_flat, targets)
        else:
            output = model(data)
            loss = criterion(output, targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=p.grad.data.pow(2)*len(data)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/dataset_size
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    print('time spent on computing fisher:',time.time()-start_time)
    return fisher

def criterion_ewc(global_model, model, fisher, output, targets, criterion, lamb=5000):
    model_old = deepcopy(global_model)
    model_old.eval()
    for param in model_old.parameters():# Freeze the weights
        param.requires_grad = False
    # Regularization for all previous tasks
    loss_reg=0
    for (name,param),(_,param_old) in zip(model.named_parameters(),model_old.named_parameters()):
        loss_reg+=torch.sum(fisher[name]*(param_old-param).pow(2))/2
    return criterion(output, targets)+lamb*loss_reg

def criterion_kd(helper, outputs, targets, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = helper.alpha
    T = helper.temperature
    KD_loss = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(outputs/T, dim=1),
                             torch.nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              torch.nn.functional.cross_entropy(outputs, targets) * (1. - alpha)
    return KD_loss


def test(helper, data_source, model):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    correct_class = np.zeros(10)
    correct_class_acc = np.zeros(10)
    correct_class_size = np.zeros(10)
    total_test_words = 0.0
    if helper.data_type == 'text':
        if helper.multi_gpu:
            hidden = model.module.init_hidden(helper.params['test_batch_size'])
        else:
            hidden = model.init_hidden(helper.params['test_batch_size'])
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        if helper.partial_test:
            data_iterator = random.sample(data_iterator, len(data_iterator)//helper.partial_test)
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            if helper.data_type == 'text':
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, helper.n_tokens)
                total_loss += len(data) * torch.nn.functional.cross_entropy(output_flat, targets).data
                hidden = tuple([each.data for each in hidden])
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
            else:
                output = model(data)
                total_loss += torch.nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                for i in range(10):
                    class_ind = targets.data.view_as(pred).eq(i*torch.ones_like(pred))
                    correct_class_size[i] += class_ind.cpu().sum().item()
                    correct_class[i] += (pred.eq(targets.data.view_as(pred))*class_ind).cpu().sum().item()
        if helper.data_type == 'text':
            acc = 100.0 * (correct / total_test_words)
            total_l = total_loss.item() / (dataset_size-1)
            if helper.multi_gpu:
                modelname = model.module.name
            else:
                modelname = model.name
            print('___Global_Test {}, Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%) | per_perplexity {:8.2f}'
                        .format(modelname, total_l, correct, total_test_words, acc, math.exp(total_l) if total_l < 30 else -1.))
            acc = acc.item()
            return total_l, acc, total_l
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            for i in range(10):
                correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
            total_l = total_loss / dataset_size
            print(f'___Test {model.name} , Average loss: {total_l},  '
                        f'Accuracy: {correct}/{dataset_size} ({acc}%)')
            return total_l, acc, correct_class_acc