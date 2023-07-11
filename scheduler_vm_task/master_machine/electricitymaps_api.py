import requests
import json

def get_24h_history(zone):
    url = "https://api-access.electricitymaps.com/2w97h07rvxvuaa1g/carbon-intensity/history"
    headers = {
    "auth-token": "e8wdUjXmhGg36i3fISwva0T2kJ35RGH0",  
    }

    response = requests.get(url, headers=headers, params={'zone' : zone,})
    data = json.loads(response.text)
    emission =[h['carbonIntensity'] for h in data['history']]

    return emission
