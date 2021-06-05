import glob 
import os
import pandas as pd 
from sklearn import model_selection
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences


def make_data():
	all_filenames = glob.glob('../crawler/data/*.csv')
	#combine all files in the list
	combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ],ignore_index = True)
	#export to csv
	combined_csv.to_csv( "../crawler/total.csv")


def split_data(paths,targets):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(paths, targets, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
	return X_train,y_train,X_val,y_val,X_test,y_test


def get_data(data_path = "../crawler/total.csv"):
	df = pd.read_csv(data_path)
	# df = df[['text','star']]
	# data = df.values.tolist()
	texts = df['text'].tolist()
	encode_text = []
	for text in texts:
		text = bpe.encode(' '.join(rdrsegmenter.tokenize(text)[0]))
		encode_ = vocab.encode_line('<s> ' + text + ' </s>',append_eos=True, add_if_not_exist=False).long().tolist()
		encode_text.append(encode_)
	stars= df['star'].tolist()
	encode_text = pad_sequences(encode_text, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

	X_train,y_train,X_val,y_val,X_test,y_test = split_data(encode_text,stars)
	return X_train,y_train,X_val,y_val,X_test,y_test


MAX_LEN = 125

rdrsegmenter = VnCoreNLP("transformers/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="transformers/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("transformers/PhoBERT_base_transformers/dict.txt")


if __name__ == '__main__':
	#make_data()
	# df = pd.read_csv("../crawler/total.csv")
	# print(df.columns.tolist())
	# print(df['category'].unique().tolist())
	# print(df.head(5))
	#get_data()
	X_train,y_train,X_val,y_val,X_test,y_test = get_data()
	text = X_train[0]
	print(text)
	print(y_train[0])
