import argparse
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter
from utils.image_helper import ImageHelper
from utils.text_helper import TextHelper
import yaml
import time
from utils.utils import *
from utils.text_load import *

logger = logging.getLogger("logger")
criterion = torch.nn.CrossEntropyLoss()


def train(helper, train_data_sets, local_model, target_model):
    """
    Performs one round of training federated `target_model`. It sequentially processes
    `train_data_sets` and returns a sum of all the models.

    :param helper: helper file with configs and useful functions
    :param train_data_sets: a subset of participant's data
    :param local_model: empty model that will be rewritten every time
    :param target_model: a global model at the previous round
    :return:
    """

    # Accumulate weights for all participants.
    weight_accumulator = dict()

    for name, data in target_model.state_dict().items():
        # don't scale tied and modified weights:
        if helper.tied and name == 'decoder.weight' or '__'in name:
            continue
        if helper.aggregation_type == 'averaging':
            weight_accumulator[name] = torch.zeros_like(data)
        else:
            # used  for median aggregation
            weight_per_model = list(data.shape)
            weight_accumulator[name] = torch.zeros([helper.no_models] + weight_per_model)

    # This is for calculating distances for Differential privacy
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    for model_id in tqdm(range(helper.no_models)):
        model = local_model
        ### copy all parameters from the target_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.lr,
                                    momentum=helper.momentum,
                                    weight_decay=helper.decay)
        model.train()
        if helper.data_type == 'text':
            current_data_model, train_data_all = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.batch_size)
            trunk = len(train_data_all)//100*(100-helper.local_test_perc)
            train_data = train_data_all[:trunk]
            test_data = train_data_all[trunk:]
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
            
        for internal_epoch in range(1, helper.retrain_no_times + 1):
            model.train()
            total_loss = 0.
            
            if helper.data_type == 'text':
                data_iterator = range(0, train_data.size(0) - 1, helper.bptt)
            else:
                data_iterator = train_data
            batch_num = 0
            for batch_id, batch in enumerate(data_iterator):
                batch_num += 1
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
                    if model_norm > helper.s_norm:
                        norm_scale = helper.s_norm / (model_norm)
                        for name, layer in model.named_parameters():
                            #### don't scale tied weights:
                            if helper.tied and name == 'decoder.weight' or '__'in name:
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


        ### sum up the model updates
        for name, data in model.state_dict().items():
            if helper.tied and name == 'decoder.weight' or '__'in name:
                continue
            if helper.aggregation_type == 'averaging':
                weight_accumulator[name].add_(data - target_model.state_dict()[name])
            else:
                weight_accumulator[name][model_id].add_(data.cpu() - target_model.state_dict()[name].cpu())

    return weight_accumulator


def eval_(helper, data_source, model):
    model.eval()
    total_loss = 0.0
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
                total_loss += len(data) * criterion(output_flat, targets).data
                hidden = helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float).item()
                total_test_words += targets.data.shape[0]
            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        if helper.data_type == 'text':
            acc = 100.0 * (correct / total_test_words)
            total_l = total_loss.item() / (dataset_size-1)
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
    return total_l, correct, total_test_words, acc


def test_local(helper, train_data_sets, target_model):
    Test_local_Acc = list()
    for model_id in range(len(train_data_sets)):
        model = target_model
        model.eval()
        if helper.data_type == 'text':
            current_data_model, train_data_all = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.batch_size)
            trunk = len(train_data_all)//100*(100-helper.local_test_perc)
            train_data = train_data_all[:trunk]
            test_data = train_data_all[trunk:]
        else:
            _, (current_data_model, test_data) = train_data_sets[model_id]
        
        local_loss, local_correct, local_total_test_wors, local_acc = eval_(helper, test_data, model)
        Test_local_Acc.append(local_acc)
    np.save('/home/ty367/federated/data/CIFAR_Test_local_Acc_first_train_averaging_diff.npy',Test_local_Acc) 


def test(helper, data_source,
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

            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        if helper.data_type == 'text':
            acc = 100.0 * (correct / total_test_words)
            total_l = total_loss.item() / (dataset_size-1)
            logger.info('___Global_Test {} poisoned: {}, Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%) | per_perplexity {:8.2f}'
                        .format(model.name, is_poison, total_l, correct, total_test_words, acc, math.exp(total_l) if total_l < 30 else -1.))
            acc = acc.item()
#             total_l = total_l.item()
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            logger.info(f'___Test {model.name} , Average loss: {total_l},  '
                        f'Accuracy: {correct}/{dataset_size} ({acc}%)')

    return total_l, acc


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
        runner_helper = ImageHelper(current_time=current_time, params=params_loaded,
                                    name=params_loaded.get('name', 'image'))
    else:
        runner_helper = TextHelper(current_time=current_time, params=params_loaded,
                                   name=params_loaded.get('name', 'text'))

    runner_helper.load_data()
    runner_helper.create_model()

    best_loss = float('inf')

    # configure logging
    if runner_helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{runner_helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.warning(f'Logging things. current path: {runner_helper.folder_path}')

        runner_helper.params['tb_name'] = args.name
        with open(f'{runner_helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(runner_helper.params, f)
    else:
        logger = create_logger()

    # setup tensorboard
    if runner_helper.tb:
        wr = SummaryWriter(log_dir=f'/home/ty367/federated/runs/{args.name}')
        runner_helper.writer = wr
        table = create_table(runner_helper.params)        
        runner_helper.writer.add_text('Model Params', table)
        print(table)

    # fix random seed
    if not runner_helper.random:
        runner_helper.fix_random()

    participant_ids = range(len(runner_helper.train_data))
    mean_acc = list()

    # save parameters
    with open(f'{runner_helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(runner_helper.params, f)

    if not runner_helper.only_eval:
        dist_list = list()
        test_loss = list()
        test_acc = list()
        
        for round in range(runner_helper.start_epoch, runner_helper.total_rounds + 1):
            start_time = time.time()

            subset_data_chunks = random.sample(participant_ids[1:], runner_helper.no_models)
            logger.info(f'Selected models: {subset_data_chunks}')
            t = time.time()
            train_data_sets = [(pos, runner_helper.train_data[pos]) for pos in subset_data_chunks]
            weight_acc = train(helper=runner_helper,
                               train_data_sets=train_data_sets,
                               local_model=runner_helper.local_model, 
                               target_model=runner_helper.target_model)
            logger.info(f'time spent on training: {time.time() - t}')
            
            # Aggregate the models
            if runner_helper.aggregation_type == 'averaging':
                runner_helper.average_shrink_models(target_model=runner_helper.target_model,
                                         weight_accumulator=weight_acc)
            elif runner_helper.aggregation_type == 'median':
                runner_helper.median_aggregation(target_model=runner_helper.target_model,
                                         weight_accumulator=weight_acc)
            else:
                raise NotImplemented(f'Aggregation {runner_helper.aggregation_type} not yet implemented.')
            
            if round in runner_helper.save_on_rounds or (round+1)%1000==0:
                t = time.time()
                logger.info(f'testing global model at round: {round}')
                round_loss, round_acc = test(helper=runner_helper,
                                             data_source=runner_helper.test_data,
                                             model=runner_helper.target_model,
                                             is_poison=False, visualize=True)
                test_loss.append(round_loss)
                test_acc.append(round_acc)
                logger.info(f'time spent on testing: {time.time() - t}')
                
                runner_helper.save_model(round=round, val_loss=round_loss)
                
            logger.info(f'Done in {time.time()-start_time} sec.')
        logger.info(f"All Test_Loss during training: {test_loss}, All Test_Acc during training: {test_acc}.")
    
    logger.info(f"start test all local models")
    if runner_helper.data_type == 'text':
        test_local(helper=runner_helper, train_data_sets=[(pos, runner_helper.train_data[pos]) for pos in
                                                    participant_ids[1:]], target_model=runner_helper.target_model)
    else:
        test_local(helper=runner_helper, train_data_sets=[(pos, runner_helper.test_local_data[pos]) for pos in
                                                    participant_ids[1:]], target_model=runner_helper.target_model)
    logger.info(f"finish test all local models, start test global model")
    final_loss, final_acc = test(helper=runner_helper, data_source=runner_helper.test_data,
                                             model=runner_helper.target_model, is_poison=False, visualize=True)
    logger.info(f"Final Test_Loss of Global model: {final_loss}, Final Test_Acc of Global model: {final_acc}.")
    logger.info(f"This run has a label: {runner_helper.params['current_time']}. ")
