import multiprocessing as mp
import pandas as pd
from selenium import webdriver
from time import sleep
import config
import json
from utils import access_url
import argparse


def crawler(cate_name,item_links):
	driver = webdriver.Chrome()
	fb_li = []
	counter = 0
	for item_link in item_links:
		access_url(driver,item_link.replace('now','foody'))
		fb_n = 1
		while True:
			while True:
				try:
					more_fb_button = driver.find_element_by_xpath(config.more_fb_bt.format('/'.join(item_link.split('/')[3:])))
					more_fb_button.click()
				except:
					break
			try:
				dic = {}
				dic['category'] = cate_name
				dic['text'] = driver.find_element_by_xpath(config.text_element.format('/'.join(item_link.split('/')[3:]),fb_n)).text
				dic['star'] = driver.find_element_by_xpath(config.star_element.format('/'.join(item_link.split('/')[3:]),fb_n)).text
				fb_li.append(dic)
				df = pd.DataFrame(fb_li)
				df.to_csv('./_data/{}.csv'.format(cate_name))
				counter += 1
			except:
				break
			fb_n += 1
	print(counter)


def multiprocess(data):
	parser = argparse.ArgumentParser(description='Multiprocessing!!!')
	parser.add_argument("-p","--processes", help="Number of processes for Multiprocessing.", type=int)
	args = parser.parse_args()
	pool = mp.Pool(args.processes)
	pool.starmap(crawler,data.items())


if __name__ == '__main__':
	with open('item_links.json','r') as f:
		data = json.load(f)
		multiprocess(data)
