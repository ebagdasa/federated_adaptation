import torch
from collections import defaultdict
from utils.helper import Helper
import random
import logging
from models.resnet import ResNet18
from torchvision import datasets, transforms
import numpy as np
logger = logging.getLogger("logger")


class ImageHelper(Helper):
    classes = None
    train_loader = None
    test_loader = None

    def create_model(self):
        local_model = ResNet18(name='Local',
                               created_time=self.params['current_time'])
        local_model.to(self.device)
        target_model = ResNet18(name='Target',
                                created_time=self.params['current_time'])
        target_model.to(self.device)
        if self.resumed_model:
            loaded_params = torch.load(f"{self.params['repo_path']}/saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_round = loaded_params['round']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current round is {self.start_round}")
        else:
            self.start_round = 1

        self.local_model = local_model
        self.target_model = target_model

    def load_data(self):
        logger.info('Loading data')

        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR10('{self.params['repo_path']}/data', train=True, download=True,
                                              transform=transform_train)

        self.test_dataset = datasets.CIFAR10('{self.params['repo_path']}/data', train=False, transform=transform_test)
        if self.recreate_dataset:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant, train_image_weight = self.sample_dirichlet_data(self.train_dataset,
                self.params['number_of_total_participants'],
                alpha=0.9)
            self.train_data = [(user, self.get_train(indices_per_participant[user])) for user in range(self.params['number_of_total_participants'])]
            self.train_image_weight = train_image_weight
            self.test_data = self.get_test()
            torch.save(self.train_data, '{self.params['repo_path']}/data/CIFAR_train_data.pt.tar')
            torch.save(self.train_image_weight, '{self.params['repo_path']}/data/CIFAR_train_image_weight.pt')
            torch.save(self.test_data, '{self.params['repo_path']}/data/CIFAR_test_data.pt.tar')
        else:
            self.train_data = torch.load('{self.params['repo_path']}/data/CIFAR_train_data.pt.tar')
            self.train_image_weight = torch.load('{self.params['repo_path']}/data/CIFAR_train_image_weight.pt')
            self.test_data = torch.load('{self.params['repo_path']}/data/CIFAR_test_data.pt.tar')

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader


    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices))
        return train_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        """
        Just mimics the TextHelper call but essentially unwraps
        batch as (data, target) tuple.

        """
        data, target = bptt
        data = data.to(self.device)
        target = target.to(self.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def sample_dirichlet_data(self, dataset, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())
        class_size = len(cifar_classes[0])
        datasize = {}
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                datasize[user, n] = no_imgs
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        train_img_size = np.zeros(no_participants)
        for i in range(no_participants):
            train_img_size[i] = sum([datasize[i,j] for j in range(10)])
        clas_weight = np.zeros((no_participants,10))
        for i in range(no_participants):
            for j in range(10):
                clas_weight[i,j] = float(datasize[i,j])/float(train_img_size[i])
        return per_participant_list, clas_weight
