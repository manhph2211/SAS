Crawling Now.vn :sleepy:
=====

# Usage 

- Following these steps:

```
git clone https://gitlab.com/edabk_nlp/crawl_data_with_selenium_manh.git
cd nowFood
python3 utils.py # getting item links of each category and save in item_links.json
mkdir _data
python3 crawler.py -p 2 # crawling comments and stars with number of processes as a option
```

# Results

```
{
    "1": 2075,
    "2": 2129,
    "3": 7557,
    "4": 5470,
    "5": 5165
}
```