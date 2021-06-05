from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch


def get_mask(data):
	masks = []
	for sen in data:
		mask = [int(token>0) for token in sen ]
		masks.append(mask)
	return masks


train_sents, train_labels, val_sents, val_labels, test_sents, test_labels = get_data() 
print(train_sents[0])
train_masks = get_mask(train_sents)
val_masks = get_mask(val_sents)

train_inputs = torch.tensor(train_sents)
val_inputs = torch.tensor(val_sents)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)

