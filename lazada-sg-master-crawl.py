#!/usr/bin/python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import json
import re
import requests
import time
from datetime import datetime, timedelta
#import logging
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.chrome.options import Options 



options = Options()
# options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('start-maximized')
options.add_argument('disable-infobars')
options.add_argument("--disable-extensions")
driver = webdriver.Chrome()


from db_id_connection3 import get_connection 
connection = get_connection()

	
ts = time.time()
date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#logging.basicConfig(filename='log/indonasia-lazada-pdp-crawl-'+date+'.log',level=logging.DEBUG)
#logging.info('Crawling starts at:'+timestamp)
#logging.debug('This message should go to the log file')
#logging.warning('And this, too')

lazada_sku = []
locations = []
sleeptime = 5
vpf_id = '1'				#$ DEFAULT
vcrawl_id = '0'				## MASTER
vlocation_id = '0'			## MASTER
vlocation_name = '0'		## MASTER
vcreated_by= 'System'	
vplatform = 'lazada'	
vcountry = 'sg'	#$ DEFAULT


def crawl_datacapture():
    					
    vpdp_title = '0'
    vweb_pid = '0'
    vpdp_url = '0'

   	
    #vpdp_page_url = 'https://shopee.co.th/shop/23325172/search?page=0&sortBy=pop'
    pdp_soup = None    
    driver.get('https://www.lazada.sg/enfagrow-official-store/?langFlag=en&q=All-Products&from=wangpu&pageTypeId=2')
    time.sleep(2)
    try:
        driver.find_element_by_xpath("//*[@id='module_age-restriction']/div/div/div[3]/div[1]").click()
        time.sleep(3)
    except:
        time.sleep(2)
    pdp_html_doc = None
    try:
        pdp_html_doc = driver.find_element_by_id("root" ).get_attribute('innerHTML')
    except NoSuchElementException:
        time.sleep(2)
    if pdp_html_doc is not None:
        pdp_soup = BeautifulSoup(pdp_html_doc, 'html.parser')
        if pdp_soup is not None:

            productSection = pdp_soup.find("div", {"class": "search-result__wrapper"}, recursive=True)
            if productSection is not None:

                productList2 = productSection.findAll("div", {"class": "search-result__wrapper"}, recursive=True)
                if productList2 is not None:
                    for product2 in productList2:
                        url = product2.find("a")
                        if url is not None:
                            vpdp_url = 'https:'+url["href"]
                            print "-=-=-=-=-=-=-=-=-"
                            print vpdp_url

                        else:
                            vpdp_url = '0'   
                        title = product2.find("div", {"class": "c16H9d"}, recursive=True)
                        
                        if title is not None:
                            title2 = title.find("a")
                            if title2 is not None:
                                vpdp_title = title2["title"]
                            
                        else:
                            vpdp_title ='0'
                        print vpdp_title


                        web_pid = product2["data-sku-simple"]
                        if web_pid is not None:
                            vweb_pid = web_pid
                            print vweb_pid
                            print "-=-=-=-=-=-=-=-=-"
                        else:
                            vweb_pid = '0'    



                        with connection.cursor() as cursor:
				    		sql1 = "INSERT INTO `lazada_sg_master` (`pdp_urls`,`platform`,`country`,`title`,`web_pid`) VALUES (%s, %s, %s, %s, %s)"
				    		try:
				    			cursor.execute(sql1, (vpdp_url, vplatform, vcountry, vpdp_title, vweb_pid))
				    			connection.commit()
				    		except cursor.Error as e:
				    			e=str(e)
				    			print(e)       

                        print('vpdp_title:: '+str(vpdp_title))
                        print('vpdp_url:: '+str(vpdp_url))
                        print('vweb_pid:: '+str(vweb_pid))
					    	
                   
		




try:
	crawl_datacapture()
except Exception as e:
	print e

finally:
	driver.quit()
	connection.close()	










               

