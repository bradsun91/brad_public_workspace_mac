import tushare as ts
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json

path_to_csv = './financial_results.csv'
base_url = 'https://www.tradingview.com'
markets_url = 'https://www.tradingview.com/markets/stocks-china/sectorandindustry-sector/'
ts.set_token('8ef5ec61cdd848715c57c11d58dd71da1271f76b2420d2bac8aef123')
pro = ts.pro_api()

# get sectors
response = requests.get(markets_url)
soup = BeautifulSoup(response.text, 'html.parser')

sectors_table = soup.find('table', {'class': 'tv-data-table'})
sectors = sectors_table.find_all('a', {'class': 'tv-screener__symbol'})

# getting company name using the API

def get_largets_company_name(sector):
    filters = {
       "filter":[
          {
             "left":"sector",
             "operation":"equal",
             "right": sector
          }
       ],
       "options":{
          "lang":"en"
       },
       "columns":[
          "name",
          "volume",
          "type",
          "subtype"
       ],
       "sort":{
          "sortBy":"volume",
          "sortOrder":"desc"
       },
       "range":[0,1]
    }
    resp = requests.post('https://scanner.tradingview.com/china/scan', data=json.dumps(filters))
    json_data = resp.json()
    return json_data['data'][0]['d'][0]

def get_finacial_info(company_name):
    # try SH market
    ts_code = f'{company_name}.SH'
    data_frame = pro.query('fina_indicator', ts_code=ts_code, start_date='20200101', end_date='20200828')
    if len(data_frame):
        return data_frame

    # try SZ market
    ts_code = f'{company_name}.SZ'
    data_frame = pro.query('fina_indicator', ts_code=ts_code, start_date='20200101', end_date='20200828')
    if len(data_frame):
        return data_frame

    return None

# Start to get the info

if __name__ == '__main__':
    df_results = []

    for sector in sectors:
        sector_name = sector.get_text()
        company_name = get_largets_company_name(sector_name)
        financial_info_df = get_finacial_info(company_name)
        df_results.append(financial_info_df)

    final_df = pd.concat(df_results)

    # Export dataframe to CSV
    final_df.to_csv(path_to_csv, encoding='utf-8', index=False)
