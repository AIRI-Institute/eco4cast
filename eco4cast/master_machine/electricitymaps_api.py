import requests
import json
from tqdm import tqdm
from time import sleep

def get_24h_history(zone, api_key):
    url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/history"
    headers = {
    "auth-token": api_key,  
    }


    while True:
        try:
            response = requests.get(url, headers=headers, params={'zone' : zone,}, timeout=5)
        except requests.exceptions.ReadTimeout:
            response = requests.get(url, headers=headers, params={'zone' : zone,}, timeout=10)
        
        if response.status_code == 404:
            raise ConnectionError('Error 404. Please check your API key and zone')
        data = json.loads(response.text)
        if 'history' in data:
            emission =[h['carbonIntensity'] for h in data['history']]
            break
        else:
            sleep(0.05)

    return emission

