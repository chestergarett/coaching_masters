import os
import requests
import pandas as pd
from dotenv import load_dotenv
from constants.constants import INPUT_PATH
load_dotenv()

def get_api_data(minDate,maxDate):
    api_url = os.getenv('URL')
    api_key = os.getenv('API_KEY')
    
    headers = {
    'x-api-key': api_key,
    'Content-Type': 'application/json'
    }

    params = {
        'minDate': minDate,
        'maxDate': maxDate
    }

    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(f'{INPUT_PATH}/input.csv',index=False)
        return df
    else:
        print(f"Request failed with status code {response.status_code}")
    
if __name__=='__main__':
    minDate = '2023-01-01'
    maxDate = '2024-10-30'
    df = get_api_data(minDate,maxDate)