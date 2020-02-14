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
        

def adapt_local(helper, train_data_sets, fisher, target_model, local_model, adaptedmodel_local_acc):    
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

                if helper.data_type == 'text':
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
            _, _, correct_class_acc = test(helper=helper, data_source=helper.test_data, model=model)
            adaptedmodel_local_acc.append((correct_class_acc*image_trainset_weight).sum())
        logger.info(f'time spent on testing: {time.time() - t}')
        if (model_id+1)%100==0 or (model_id+1)==len(train_data_sets):
            logger.info(f'Saved adaptedmodel_local_acc at model_id: {model_id}')
            np.save(helper.save_name + '_AdaptedModel_LocalTest_Acc.npy',np.array(adaptedmodel_local_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default=f'{adaptation_helper.repo_path}/utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params_loaded = yaml.load(f)

    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['data_type'] == "image":
        adaptation_helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image_adapt'))
    else:
        adaptation_helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text_adapt'))

    adaptation_helper.load_data()
    adaptation_helper.create_model()

    # configure logging
    if adaptation_helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{adaptation_helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.warning(f'Logging things. current path: {adaptation_helper.folder_path}')
        adaptation_helper.params['tb_name'] = args.name
        with open(f'{adaptation_helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(adaptation_helper.params, f)
    else:
        logger = create_logger()

    if adaptation_helper.tb:
        wr = SummaryWriter(log_dir=f'{adaptation_helper.repo_path}/runs/{args.name}')
        adaptation_helper.writer = wr
        table = create_table(helper.params)
        adaptation_helper.writer.add_text('Model Params', table)
        print(adaptation_helper.lr, table)

    if not adaptation_helper.random:
        adaptation_helper.fix_random()

    participant_ids = range(len(adaptation_helper.train_data))
    mean_acc = list()
    
    # save parameters:
    with open(f'{adaptation_helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(adaptation_helper.params, f)
    if not adaptation_helper.only_eval:
        if adaptation_helper.ewc:
            if not os.path.exists(adaptation_helper.resumed_fisher):
                fisher = fisher_matrix_diag(adaptation_helper, adaptation_helper.test_data, adaptation_helper.target_model, criterion)
                torch.save(fisher, adaptation_helper.resumed_fisher)
            else:
                fisher = torch.load(adaptation_helper.resumed_fisher)
        else:
            fisher = None
        random.seed(66)
        if not os.path.exists(adaptation_helper.save_name + '_AdaptedModel_LocalTest_Acc.npy'):
            adaptedmodel_local_acc = list()
        else:
            adaptedmodel_local_acc = list(np.load(adaptation_helper.save_name + '_AdaptedModel_LocalTest_Acc.npy'))
        subset_data_chunks = participant_ids[len(adaptedmodel_local_acc):]
        logger.info(f'Selected adapted models ID: {subset_data_chunks}')
        t1 = time.time()
        adapt_local(helper=adaptation_helper, train_data_sets=[(pos, adaptation_helper.train_data[pos]) for pos in subset_data_chunks], fisher=fisher, target_model=adaptation_helper.target_model, local_model=adaptation_helper.local_model, adaptedmodel_local_acc=adaptedmodel_local_acc)
        logger.info(f'time spent on local adaptation: {time.time() - t1}')
    logger.info(f"Evaluate the global (target) model on participants' local testdata to get local accuracies of federated learning model")
    if adaptation_helper.data_type == 'text':
        globalmodel_local_acc = test_globalmodel_local(helper=adaptation_helper, data_sets=[(pos, adaptation_helper.train_data[pos]) for pos in participant_ids], target_model=adaptation_helper.target_model)
    else:       
        _, _, globalmodel_correct_class_acc = test(helper=adaptation_helper, data_source=adaptation_helper.test_data, model=adaptation_helper.target_model)
        globalmodel_local_acc = (globalmodel_correct_class_acc*adaptation_helper.train_image_weight).sum(-1)
    np.save(adaptation_helper.save_name + '_GlobalModl_LocalTest_Acc.npy',np.array(globalmodel_local_acc))
    logger.info(f"This run has a label: {adaptation_helper.params['current_time']}. ")