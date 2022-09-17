

import time
import parsel
import csv
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()
driver.get('https://www.linkedin.com')

sleep(3)

username = driver.find_element(By.ID, 'session_key')
username.send_keys("email@email.com")

sleep(3)

password = driver.find_element(By.ID, 'session_password')
password.send_keys('XXXXX')

sleep(3)

login = driver.find_element(By.CLASS_NAME, 'sign-in-form__submit-button')
login.click()

sleep(15)

search_term = driver.find_element(By.CLASS_NAME, 'global-nav-search')
search_term.send_keys("Data Scientist")

sleep(15)

