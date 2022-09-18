## Import necessary libraries
import time
import parsel
import csv
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

## Initialize chrome webdriver 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()
driver.get('https://www.linkedin.com')

sleep(2)    ## I use sleep throughout the script to allow for webpage loading lags

## Login to a linkedin account
username = driver.find_element(By.ID, 'session_key')
username.send_keys("eugenebaraka@gmail.com")

sleep(2)

password = driver.find_element(By.ID, 'session_password')
password.send_keys('Tuamini128.')

sleep(2)

login = driver.find_element(By.CLASS_NAME, 'sign-in-form__submit-button')
login.click()

sleep(15)

search_term = driver.find_element(By.ID, "ember28")
search_term.send_keys("Data Scientist")
search_term.send_keys(Keys.RETURN)

# # sleep(15)

# ## Search profiles using google





# search_profile = driver.find_element(By.CLASS_NAME, 'q')
# linkedin_urls = driver.find_element(By.CSS_SELECTOR, '.r a')
# linkedin_urls = [url.get_attribute('href') for url in linkedin_urls]

# for profile in linkedin_urls:
#         search_profile.send_keys("profile AND 'Data Scientist'")
#         search_profile.send_keys(Keys.RETURN)
