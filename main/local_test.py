from dataloader import test_dataloader,get_mask
from train import flat_accuracy
from transformers import RobertaForSequenceClassification, RobertaConfig
import numpy as np 
import torch
from tqdm.notebook import tqdm
from utils import * 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


config = RobertaConfig.from_pretrained(
    "transformers/PhoBERT_base_transformers/config.json", from_tf=False, num_labels = 6, output_hidden_states=False,
)
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "transformers/PhoBERT_base_transformers/model.bin",
    config=config
)

device = 'cpu'
BERT_SA.load_state_dict(torch.load('model.pth',map_location='cpu'))
BERT_SA.eval()


def test(test_loader):
    print("Running Validation...")
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in tqdm(test_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = BERT_SA(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print(" F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))
  

def predict(text):
  #test(test_dataloader)
  # while True:
  #   text = input("Nhập:")
  text = bpe.encode(' '.join(rdrsegmenter.tokenize(text)[0]))
  encode_ = vocab.encode_line('<s> ' + text + ' </s>',append_eos=True, add_if_not_exist=False).long().tolist()
  encode_text = pad_sequences([encode_], maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

  test_masks = get_mask(encode_text)
  test_masks = torch.tensor(test_masks,dtype = torch.int64)
  test_inputs = torch.tensor(encode_text)

  test_inputs = test_inputs.to(device)
  test_masks = test_masks.to(device)
    
  with torch.no_grad():
    outputs = BERT_SA(test_inputs, token_type_ids=None, attention_mask=test_masks)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    predict = np.argmax(logits)
    
    if predict == 5:
      te = "Quá tốt"
    elif predict == 4:
      te = "Tốt"
    elif predict == 3:
      te = "Bình thường"
    elif predict == 2:
      te = "Chán"
    elif predict == 1:
      te = "Tệ v"
    return te



