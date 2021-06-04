from selenium import webdriver
from time import sleep
import config
import json


def access_url(driver,url):
    driver.get(str(url))
    sleep(config.sleep_time)


def save_categories(path,categories):
	with open(path,'w') as f:
		for cate in categories:
			f.write(str(cate)+'\n')


def get_categories(driver):
	categories = []
	i=0
	while True:
		i+=1
		try:
			a = driver.find_element_by_xpath(config.cateXpath.format(i))
			categories.append(a.get_attribute('href'))
		except:
			break
	save_categories(config.category_path,categories)


def get_cate_name(cate):
	cate = cate.split('-')
	cate = cate[7:-3]
	return ' '.join(cate)


def load_cates(path=config.category_path):
	categories = []
	with open(path,'r') as f:
		for cate in f.readlines():
			categories.append(cate[:-1])
	return categories[1:]


def get_item_links(driver,categories):
	dic = {}
	for cate in load_cates():
		cate_name = get_cate_name(cate)
		item_links = []
		access_url(driver,cate)
		page_n = 0
		while True:
			page_n+=1
			try:
				page = driver.find_element_by_xpath(config.page_xpath.format(page_n))
				page.click()
				sleep(config.sleep_time/2)
				i=1
				while True:
					i+=1
					try:
						a = driver.find_element_by_xpath(config.item_xpath.format(i))
						item_links.append(a.get_attribute('href'))
						sleep(config.sleep_time)
					except:
						break
			except:
				break
		dic[cate_name]=item_links

	with open('item_links.json','w') as f:
		json.dump(dic,f,indent=4)



if __name__ == '__main__':
	driver = webdriver.Chrome()
	access_url(driver,config.main_web_url)
	get_categories(driver)
	categories = load_cates()
	get_item_links(driver,categories)


