from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 


def get_mask(data):
	masks = []
	for sen in data:
		mask = [int(token>0) for token in sen ]
		masks.append(mask)
	return masks


train_sents, train_labels, val_sents, val_labels, test_sents, test_labels = get_data() 

train_masks = get_mask(train_sents)
val_masks = get_mask(val_sents)
test_masks = get_mask(test_sents)

test_masks = torch.tensor(test_masks,dtype = torch.int64)
test_inputs = torch.tensor(test_sents)
test_labels = torch.tensor(test_labels, dtype = torch.int64)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)

train_inputs = torch.tensor(train_sents)
train_labels = torch.tensor(train_labels,dtype = torch.int64)
train_masks = torch.tensor(train_masks,dtype = torch.int64)

val_masks = torch.tensor(val_masks,dtype = torch.int64)
val_inputs = torch.tensor(val_sents)
val_labels = torch.tensor(val_labels, dtype = torch.int64)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=8)

