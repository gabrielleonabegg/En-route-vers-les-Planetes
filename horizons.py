import json
import requests
import datetime
import numpy as np
from general_definitions import pEpoch

# https://stackoverflow.com/questions/100210/what-is-the-standard-way-to-add-n-seconds-to-datetime-time-in-python

def horizons(id, dtime=None):
  # Define API URL and SPK filename:
  url = 'https://ssd.jpl.nasa.gov/api/horizons.api'

  # Define the time span:

  if dtime is None:
    epoch = datetime.datetime.now()
    start_time = epoch.strftime('%Y-%b-%d')
    start_time += "%2000:00:00"
    stop_time = epoch.strftime('%Y-%b-%d') + "%2000:15"
  elif dtime!=0:
    epoch = pEpoch()
    start_time_d = (epoch + datetime.timedelta(0,dtime))
    start_time = start_time_d.strftime('%Y-%b-%d') + "%20" + start_time_d.strftime('%H:%M:%S')
    stop_time_d = (epoch + datetime.timedelta(0,dtime)  + datetime.timedelta(0,900))
    stop_time = stop_time_d.strftime('%Y-%b-%d') + "%20" + stop_time_d.strftime('%H:%M:%S')
  elif dtime == 0:
    epoch = pEpoch()
    start_time = epoch.strftime('%Y-%b-%d')
    start_time += "%2000:00:00"
    stop_time = epoch.strftime('%Y-%b-%d') + "%2000:15"


  # Build the appropriate URL for this API request:
  # IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
  #            Horizons COMMAND parameter specification.
  url += "?format=json&OBJ_DATA=NO&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='%400'&VEC_TABLE='2x'&CAL_TYPE='GREGORIAN'&CSV_FORMAT=YES&TIME_DIGITS='SECONDS'"
  url += "&START_TIME='{}'&STOP_TIME='{}'".format(start_time, stop_time) 
  url += "&COMMAND='{}'".format(id)

  # Submit the API request and decode the JSON-response:
  response = requests.get(url)
  try:
    data = json.loads(response.text)
  except ValueError:
    print("Unable to decode JSON results")

  # If the request was valid...
  if (response.status_code == 200):
    data = data["result"]
    # print(data)
    s = data.find("$$SOE")
    e = data.find("n.a.,")
    data = data[s + 6:e].split(', ')
    data = data[2:8]
    data = list(map(str.lower, data))
    r = np.asarray(data[:3], dtype=np.longdouble)
    v = np.asarray(data[3:], dtype=np.longdouble)
    return r*1000, v*1000

  # If the request was invalid, extract error content and display it:
  if (response.status_code == 400):
    data = json.loads(response.text)
    if "message" in data:
      print("MESSAGE: {}".format(data["message"]))
    else:
      print(json.dumps(data, indent=2))

  # Otherwise, some other error occurred:
  print("response code: {0}".format(response.status_code))
  return None
