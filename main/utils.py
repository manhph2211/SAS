import glob 
import os
import pandas as pd 
from model import TF_IDF
from sklearn import model_selection


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


def make_vocab(words):
	pass



def get_data(data_path = "../crawler/total.csv"):
	df = pd.read_csv(data_path)
	# df = df[['text','star']]
	# data = df.values.tolist()
	texts = df['text'].tolist()
	stars= df['star'].tolist()
	X_train,y_train,X_val,y_val,X_test,y_test = split_data(texts,stars)
	return X_train,y_train,X_val,y_val,X_test,y_test


if __name__ == '__main__':
	#make_data()
	# df = pd.read_csv("../crawler/total.csv")
	# print(df.columns.tolist())
	# print(df['category'].unique().tolist())

	# print(df.head(5))
	#get_data()
	X_train,y_train,X_val,y_val,X_test,y_test = get_data()
	tfidf = TF_IDF(X_train)
	words = tfidf.get_vocab()
	print(words)