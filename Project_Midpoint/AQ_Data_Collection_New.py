#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import base64
enc = base64.b64encode
import csv
import json

#site = 'https://app.cpcbccr.com/caaqms/fetch_table_data'
site = 'https://app.cpcbccr.com/caaqms/advanced_search'

from_date = input("Enter From Date (dd-mm-yyyy): ").encode("utf-8")
to_date = input("Enter To Date (dd-mm-yyyy): ").encode("utf-8")

decoded_s1 = b'{"criteria":"1 Hours","recd portFormat":"Tabular","fromDate":"'+from_date+b' T00:00:00Z","toDate":"'+to_date+b' T00:00:59Z","state":"Delhi","city":"Delhi","station":"site_'

decoded_s2 = b'","parameter":["parameter_215","parameter_193","parameter_204","parameter_238","parameter_237","parameter_235","parameter_234","parameter_236","parameter_226","parameter_225","parameter_194","parameter_311","parameter_312","parameter_203","parameter_222","parameter_202","parameter_232","parameter_223","parameter_240","parameter_216"],"parameterNames":["PM10","PM2.5","AT","BP","SR","RH","WD","RF","NO","NOx","NO2","NH3","SO2","CO","Ozone","Benzene","Toluene","Xylene","MP-Xylene","Eth-Benzene"]}'

#station_codes = [b'5024',b'301',b'1420',b'108', b'1560', b'104', b'103', b'118', b'1421', b'1422', b'116', b'106', b'114', b'117', b'1423', b'1424', b'109', b'1425', b'122', b'1561']
station_codes = [b'1563']
print(len(station_codes))

parameters = ['PM10', 'PM2.5', 'AT', 'BP', 'SR', 'RH', 'WD', 'RF', 'NO', 'NOx', 'NO2', 'NH3', 'SO2', 'CO', 'Ozone', 'Benzene', 'Toluene', 'Xylene', 'MP-Xylene', 'Eth-Benzene']

#Schema: {Place, from_date, time, to_date, time, 'PM10', 'PM2.5', 'AT', 'BP', 'SR', 'RH', 'WD', 'RF', 'NO', 'NOx', 'NO2', 'NH3', 'SO2', 'CO', 'Ozone', 'Benzene', 'Toluene', 'Xylene', 'MP-Xylene', 'Eth-Benzene'}

for code in station_codes:
    file = open("AQ_1563_Aug21_July22_Neww.csv","a+", newline="")
    w = csv.writer(file)
    payload = str(enc(decoded_s1+code+decoded_s2),"utf-8")
    r = requests.post(site,payload,verify=False)
    if r.status_code!=200:
       print("Error")
       continue
    s1 = r.text
    y = json.loads(s1)
    place = y['tabularData']['title']
    x = y['tabularData']['bodyContent']
    print(len(x))
    for i in range(len(x)):
        from_time = x[i]['from date'].split()
        to_time = x[i]['to date'].split()
        arr = [x[i][j] for j in parameters]
        w.writerow([place,from_time[0],from_time[2],to_time[0],to_time[2]]+arr)
    file.close()
    print(place+" ---- Done!")
    


# 

# In[ ]:




