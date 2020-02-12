import argparse
import json
from datetime import datetime
import os
import logging
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper

from utils.utils import dict_html

logger = logging.getLogger("logger")
import yaml
import time
import numpy as np

import random
from utils.utils import *
from utils.text_load import *
from copy import deepcopy


criterion = torch.nn.CrossEntropyLoss()

def eval_one_participant(helper, data_source, model):
    model.eval()
    correct = 0.0
    total_test_words = 0.0
    if helper.data_type == 'text':
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source
    
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            if helper.data_type == 'text':
                hidden = model.init_hidden(data.size(-1))
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, helper.n_tokens)
                hidden = helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float).item()
                total_test_words += targets.data.shape[0]
            else:
                output = model(data)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        if helper.data_type == 'text':
            acc = 100.0 * (correct / total_test_words)
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
    return acc

def test_globalmodel_local(helper, data_sets, target_model):
    globalmodel_local_acc = list()
    for model_id in range(len(data_sets)):
        model = target_model
        model.eval()
        if helper.data_type == 'text':
            current_data_model, train_data_all = data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            if helper.multi_gpu:
                hidden = model.module.init_hidden(helper.batch_size)
            else:
                hidden = model.init_hidden(helper.batch_size)
            trunk = len(train_data_all)//100*(100-helper.local_test_perc)
            test_data = train_data_all[trunk:] # only take the last 10% as test set
        else:
            _, (current_data_model, test_data) = data_sets[model_id]
        local_acc = eval_one_participant(helper, test_data, model)
        globalmodel_local_acc.append(local_acc)
    return globalmodel_local_acc

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
                total_loss += len(data) * criterion(output_flat, targets).data
                hidden = tuple([each.data for each in hidden])
#                 hidden = helper.repackage_hidden(hidden) # manually do this in last line
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
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
            logger.info('___Global_Test {}, Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%) | per_perplexity {:8.2f}'
                        .format(modelname, total_l, correct, total_test_words, acc, math.exp(total_l) if total_l < 30 else -1.))
            acc = acc.item()
            return acc
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            for i in range(10):
                correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
            total_l = total_loss / dataset_size
            logger.info(f'___Test {model.name} , Average loss: {total_l},  '
                        f'Accuracy: {correct}/{dataset_size} ({acc}%)')
            return correct_class_acc
        

def adapt_local(adaptedmodel_local_acc, fisher, helper, train_data_sets, local_model, target_model):    
    for parame in target_model.parameters():
        parame.requires_grad = False
    for model_id in tqdm(range(len(train_data_sets))):
        iteration = 0
        model = local_model
        if not helper.scratch:
            model.copy_params(target_model.state_dict())
        if helper.multi_gpu:
            model = torch.nn.DataParallel(model, dim=1).cuda()
        if helper.freeze_base:
            if helper.data_type == 'text':
                freeze = 4
            else:
                freeze = 60
            num = 0
            for parame in model.parameters():
                if num < freeze:
                    parame.requires_grad = False
                num += 1
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=helper.lr,
                                        momentum=helper.momentum,
                                        weight_decay=helper.decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=helper.lr,
                                        momentum=helper.momentum,
                                        weight_decay=helper.decay)
        model.train()
        if helper.data_type == 'text':
            current_data_model, train_data_all = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            if helper.multi_gpu:
                hidden = model.module.init_hidden(helper.batch_size)
            else:
                hidden = model.init_hidden(helper.batch_size)
            trunk = len(train_data_all)//100*(100-helper.local_test_perc)
            train_data = train_data_all[:trunk]
            test_data = train_data_all[trunk:]
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
            image_trainset_weight = np.zeros(10)
            for ind, x in enumerate(train_data):
                _, label = x
                for labeli in range(10):
                    image_trainset_weight[labeli] += (label==labeli).sum()
            image_trainset_weight = image_trainset_weight/image_trainset_weight.sum()
        
        start_time = time.time()
        for internal_epoch in range(1, helper.retrain_no_times + 1):
            model.train()            
            if helper.data_type == 'text':
                data_iterator = range(0, train_data.size(0) - 1, helper.bptt)
            else:
                data_iterator = train_data
            batch_num = 0                
            for batch_id, batch in enumerate(data_iterator):
                iteration += 1
                batch_num += 1
                optimizer.zero_grad()
                data, targets = helper.get_batch(train_data, batch,
                                                  evaluation=False)
                if helper.data_type == 'text':
                    hidden = tuple([each.data for each in hidden])
                    if helper.kd:
                        with torch.no_grad():
                            teacher_outputs, _ = target_model(data, hidden)
                    output, hidden = model(data, hidden)
                    if helper.ewc:
                        loss = criterion_ewc(target_model, model, fisher, output.view(-1, ntokens), targets, criterion, lamb=helper.lamb)
                    elif helper.kd:
                        loss = criterion_kd(helper, output.view(-1, ntokens), targets, teacher_outputs.view(-1, ntokens))
                    else:
                        loss = criterion(output.view(-1, ntokens), targets)
                else:
                    output = model(data)
                    if helper.ewc:
                        loss = criterion_ewc(target_model, model, fisher, output, targets, criterion, lamb=helper.lamb)
                    elif helper.kd:
                        with torch.no_grad():
                            teacher_outputs = target_model(data)
                        loss = criterion_kd(helper, output, targets, teacher_outputs)
                    else:
                        loss = criterion(output, targets)
                loss.backward()

                if helper.diff_privacy:
                    optimizer.step()
                    model_norm = helper.model_dist_norm(model, target_params_variables)
                    if model_norm > helper.params['s_norm']:
                        norm_scale = helper.params['s_norm'] / (model_norm)
                        for name, layer in model.named_parameters():
                            #### don't scale tied weights:
                            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                continue
                            clipped_difference = norm_scale * (
                            layer.data - target_model.state_dict()[name])
                            layer.data.copy_(
                                target_model.state_dict()[name] + clipped_difference)
                elif helper.data_type == 'text':
                    # `clip_grad_norm` helps prevent the exploding gradient
                    # problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                    optimizer.step()
                else:
                    optimizer.step()
        t = time.time()
        logger.info(f'time spent on local adaptation: {t-start_time}')
        logger.info(f'testing adapted model on local testset at model_id: {model_id}')
        if helper.data_type == 'text':
            local_acc = eval_one_participant(helper, test_data, model)
            adaptedmodel_local_acc.append(local_acc)
        else:
            epoch_loss, correct_class_acc = test(helper=helper, data_source=helper.test_data, model=model)
            adaptedmodel_local_acc.append((correct_class_acc*image_trainset_weight).sum())
        logger.info(f'time spent on testing: {time.time() - t}')
        if (model_id+1)%100==0 or (model_id+1)==len(train_data_sets):
            logger.info(f'Saved adaptedmodel_local_acc at model_id: {model_id}')
            np.save(helper.save_name + '_AdaptedModel_LocalTest_Acc.npy',np.array(adaptedmodel_local_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/adapt_image.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params_loaded = yaml.load(f)

    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['data_type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image_adapt'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text_adapt'))

    helper.load_data()
    helper.create_model()

    # configure logging
    if helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.warning(f'Logging things. current path: {helper.folder_path}')
        helper.params['tb_name'] = args.name
        with open(f'{helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(helper.params, f)
    else:
        logger = create_logger()

    if helper.tb:
        wr = SummaryWriter(log_dir=f'./runs/{args.name}')
        helper.writer = wr
        table = create_table(helper.params)
        helper.writer.add_text('Model Params', table)
        print(helper.lr, table)

    if not helper.random:
        helper.fix_random()

    participant_ids = range(len(helper.train_data))
    mean_acc = list()
    
    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    if not helper.only_eval:
        if helper.ewc:
            if not os.path.exists(helper.resumed_fisher):
                fisher = fisher_matrix_diag(helper, helper.test_data, helper.target_model, criterion)
                torch.save(fisher, helper.resumed_fisher)
            else:
                fisher = torch.load(helper.resumed_fisher)
        else:
            fisher = None
        random.seed(66)
        if not os.path.exists(helper.save_name + '_AdaptedModel_LocalTest_Acc.npy'):
            adaptedmodel_local_acc = list()
        else:
            adaptedmodel_local_acc = list(np.load(helper.save_name + '_AdaptedModel_LocalTest_Acc.npy'))
        subset_data_chunks = participant_ids[len(adaptedmodel_local_acc):]
        logger.info(f'Selected adapted models ID: {subset_data_chunks}')
        t1 = time.time()
        adapt_local(adaptedmodel_local_acc=adaptedmodel_local_acc, fisher=fisher, helper=helper, train_data_sets=[(pos, helper.train_data[pos]) for pos in subset_data_chunks],local_model=helper.local_model, target_model=helper.target_model)
        logger.info(f'time spent on local adaptation: {time.time() - t1}')
    logger.info(f"Evaluate the global (target) model on participants' local testdata to get local accuracies of federated learning model")
    if helper.data_type == 'text':
        globalmodel_local_acc = test_globalmodel_local(helper=helper, data_sets=[(pos, helper.train_data[pos]) for pos in participant_ids], target_model=helper.target_model)
    else:       
        globalmodel_correct_class_acc = test(helper=helper, data_source=helper.test_data, model=helper.target_model)
        globalmodel_local_acc = (globalmodel_correct_class_acc*helper.train_image_weight).sum(-1)
    np.save(helper.save_name + '_GlobalModl_LocalTest_Acc.npy',np.array(globalmodel_local_acc))
    logger.info(f"This run has a label: {helper.params['current_time']}. ")