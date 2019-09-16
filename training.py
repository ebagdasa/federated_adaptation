import argparse
import json
from datetime import datetime
import os
import logging
import torch
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
# logger.setLevel("ERROR")
import yaml
import time
import numpy as np

import random
from utils.utils import *
from utils.text_load import *

criterion = torch.nn.CrossEntropyLoss()

def train(helper, epoch, train_data_sets, local_model, target_model, last_weight_accumulator):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()


    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
    current_number_of_adversaries = 0

    for model_id in tqdm(range(helper.no_models)):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.lr,
                                    momentum=helper.momentum,
                                    weight_decay=helper.decay)
        model.train()

        start_time = time.time()
        if helper.data_type == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.batch_size)
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]

        for internal_epoch in range(1, helper.retrain_no_times + 1):
            total_loss = 0.
            if helper.data_type == 'text':
                data_iterator = range(0, train_data.size(0) - 1, helper.bptt)
            else:
                data_iterator = train_data


            for batch_id, batch in enumerate(data_iterator):
                optimizer.zero_grad()
                data, targets = helper.get_batch(train_data, batch,
                                                  evaluation=False)
                if helper.data_type == 'text':
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    loss = criterion(output.view(-1, ntokens), targets)
                else:
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)

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

                total_loss += loss.item()

                if helper.report_train_loss and batch_id % helper.params[
                    'log_interval'] == 0 and batch_id > 0:
                    cur_loss = total_loss.item() / helper.log_interval
                    elapsed = time.time() - start_time
                    logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'
                                        .format(model_id, epoch, internal_epoch,
                                        batch_id, train_data.size(0) // helper.bptt,
                                        helper.params['lr'],
                                        elapsed * 1000 / helper.log_interval,
                                        cur_loss,
                                        math.exp(cur_loss) if cur_loss < 30 else -1.))
                    total_loss = 0
                    start_time = time.time()


        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    return weight_accumulator


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    if helper.data_type == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
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
                hidden = helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]

                if batch_id == random_print_output_batch * helper.params['bptt'] and \
                        helper.params['output_examples'] and epoch % 5 == 0:
                    expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                    expected_sentence = f'*EXPECTED*: {expected_sentence}'
                    predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                    predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                    score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                    logger.info(expected_sentence)
                    logger.info(predicted_sentence)

                    logger.info(f"<h2>Epoch: {epoch}_{helper.params['current_time']}</h2>"
                             f"<p>{expected_sentence.replace('<','&lt;').replace('>', '&gt;')}"
                             f"</p><p>{predicted_sentence.replace('<','&lt;').replace('>', '&gt;')}</p>"
                             f"<p>Accuracy: {score} ")
            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        if helper.data_type == 'text':
            acc = 100.0 * (correct / total_test_words)
            total_l = total_loss.item() / (dataset_size-1)
            logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                           total_l, correct, total_test_words,
                                                           acc))
            acc = acc.item()
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            logger.info(f'___Test {model.name} , epoch: {epoch}: Average loss: {total_l},  '
                        f'Accuracy: {correct}/{dataset_size} ({acc}%)')

    return (total_l, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params_loaded = yaml.load(f)

    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['data_type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    helper.load_data()
    helper.create_model()

    best_loss = float('inf')

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
        wr = SummaryWriter(log_dir=f'runs/{args.name}')
        helper.writer = wr
        table = create_table(helper.params)
        helper.writer.add_text('Model Params', table)

    if not helper.random:
        helper.fix_random()


    participant_ids = range(len(helper.train_data))
    mean_acc = list()

    weight_accumulator = None

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        subset_data_chunks = random.sample(participant_ids[1:], helper.no_models)
        logger.info(f'Selected models: {subset_data_chunks}')
        t = time.time()
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model,
                                    last_weight_accumulator=weight_accumulator)
        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)
        # del weight_accumulator
        t = time.time()

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False, visualize=True)

        logger.info(f'time spent on testing: {time.time() - t}')
        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time()-start_time} sec.')

    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")
