import pandas as pd 
import glob
import json


def converter(point):
	star = 0

	if point >=8.5:
		star = 5
	elif 8.5>point>=7.5:
		star = 4
	elif 7.5>point>=6:
		star = 3
	elif 6>point>=4:
		star = 2
	else:
		star = 1


	return star


def counter():
	counter = {}
	for i in range(1,6):
		counter[i] = 0

	csv_files = glob.glob('./*.csv')
	for file in csv_files:
		df = pd.read_csv(file)
		df['star'] = df['star'].apply(lambda x: converter(x))
		stars = df['star'].to_numpy()
		for star in stars:
			counter[star] += 1
		df.to_csv(file)
	with open('./results.json','w') as f:
		json.dump(counter,f,indent=4)

if __name__ == '__main__':
	counter()