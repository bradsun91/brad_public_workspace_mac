# Create a single py file that needs to open VPN to generate the most recent sp500 tickers from wiki:
import bs4 as bs
import pickle
import requests

#import pandas as pd 

def save_sp500_tickers(sector=False):
    # make request on SP500 wiki page and find the stock table
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})

    # if sector is true, fetch industry sector with tickers
    if sector:
        # create ticker_sector set
        ticker_sector = set()
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip('\n')
            sector = row.findAll('td')[3].text.lower().replace(' ','_')
                
            # add tuple to ticker_sector set
            ticker_sector.add((ticker,sector))

        return ticker_sector

    # fetch tickers
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip('\n')
        tickers.append(ticker)
    #with open('sp500tickers.pickle','wb') as f:
        #pickle.dump(tickers,f)

    return tickers  

tickers = save_sp500_tickers(sector=False)

df = pd.DataFrame(tickers, columns=['sp500_tickers'])
data_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/quant_research/data_pipeline/"
df.to_csv(data_path+"most_recent_sp500_tickers.csv", index = False)