import requests
import json

def get_24h_history(zone, api_key):
    url = "https://api-access.electricitymaps.com/2w97h07rvxvuaa1g/carbon-intensity/history"
    headers = {
    "auth-token": api_key,  
    }

    response = requests.get(url, headers=headers, params={'zone' : zone,})
    data = json.loads(response.text)
    emission =[h['carbonIntensity'] for h in data['history']]

    return emission
