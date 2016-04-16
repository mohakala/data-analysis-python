# Scrape a html page
# http://docs.python-guide.org/en/latest/scenarios/scrape/

from lxml import html
import requests

url1='http://econpy.pythonanywhere.com/ex/001.html'
page = requests.get(url1)
tree = html.fromstring(page.content)
#This will create a list of buyers:
buyers = tree.xpath('//div[@title="buyer-name"]/text()')
#This will create a list of prices:
prices = tree.xpath('//span[@class="item-price"]/text()')
print(buyers)
print(prices)



# Read a html page
# https://urllib3.readthedocs.org/en/latest/
# http://tealscientific.com/blog/?p=2373 

import urllib3

urleq = 'http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv'
http = urllib3.PoolManager()
response = http.request('GET', urleq)
if(False):
    with open('all_week.csv', 'wb') as f:
        f.write(response.data)
    response.release_conn()
if(False): print(response.data)
        


# Parse JSON from string
# http://docs.python-guide.org/en/latest/scenarios/json/
# http://stackoverflow.com/questions/7771011/parse-json-in-python

import simplejson as json

stri='{"type":"fi.prh.opendata.bis","version":"1","totalResults":-1,"resultsFrom":0,"previousResultsUri":null,"nextResultsUri":"http://avoindata.prh.fi/opendata/bis/v1?totalResults=false&maxResults=10&resultsFrom=10&companyRegistrationFrom=2014-02-28","exceptionNoticeUri":null,"results":[{"businessId":"2755076-6","name":"Kainuun AHA-asiantuntijat Oy","registrationDate":"2016-04-02","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755076-6"},{"businessId":"2755080-3","name":"Kattotyön Tekniikka Oy","registrationDate":"2016-04-02","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755080-3"},{"businessId":"2755077-4","name":"Waccessory Oy","registrationDate":"2016-04-02","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755077-4"},{"businessId":"2755078-2","name":"Kärkitimpurit Oy","registrationDate":"2016-04-02","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755078-2"},{"businessId":"2755067-8","name":"Jesi Verkkokaupat Oy","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755067-8"},{"businessId":"2755018-4","name":"Nokkamiehet Oy","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755018-4"},{"businessId":"2754993-3","name":"Jyränoja Consulting Oy","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2754993-3"},{"businessId":"2755022-1","name":"Seloy Live Oy","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755022-1"},{"businessId":"2754876-2","name":"Hyryn Erälomat OY","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2754876-2"},{"businessId":"2755038-7","name":"SaimaanKeskus Oy","registrationDate":"2016-04-01","companyForm":"OY","detailsUri":"http://avoindata.prh.fi/opendata/bis/v1/2755038-7"}]}'
js = json.loads(stri)
print(js['version'])



# Read JSON
# http://stackoverflow.com/questions/16573332/jsondecodeerror-expecting-value-line-1-column-1-char-0

url2='http://avoindata.prh.fi:80/bis/v1?totalResults=false&maxResults=10&resultsFrom=0&companyRegistrationFrom=2014-02-28'
if(False):
    j_obj = requests.get(url2).json()
    print(j_obj['version'])
    print(j_obj['results'][0])
    print(j_obj['results'][1])
    print(j_obj['results'][1]['businessId'])
    for i in range(1,10):
        print(j_obj['results'][i]['name'])
        








