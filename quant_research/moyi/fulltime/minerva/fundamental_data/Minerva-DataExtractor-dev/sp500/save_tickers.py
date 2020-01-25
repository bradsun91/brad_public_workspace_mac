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

if __name__ == '__main__':
    '''ticker_sector = save_sp500_tickers(sector=True)
    ticker_sector = pd.DataFrame(ticker_sector, columns=['tickers','sectors'])
    ticker_sector.to_csv('ticker_sector.csv')
    print (ticker_sector)'''