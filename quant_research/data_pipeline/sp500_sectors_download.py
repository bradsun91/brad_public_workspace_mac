import requests
import bs4 as bs
import pandas as pd
from datetime import datetime


### Create a function to extract the "sector" of stocks in S&P 500
def save_sp500_sector():
    """
    This is a function to save current stock sectors in S&P500

    Input: None

    Output:
    sectors: A list of sector name of S&P500 stocks
    """
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    ### Collect the sectors Name from Wikipedia tables
    sectors = []
    for row in table.findAll('tr')[1:]:
        sector = row.findAll('td')[3].text
        sectors.append(sector)
    ### Remove the newline character from list
    sample_list = sectors
    converted_list = []
    for element in sample_list:
        converted_list.append(element.strip())
    sectors = converted_list

    return sectors


if __name__ == "__main__":
    print("Updating SP500 sectors...")
    sp500_tickers_location = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/quant_research/sp500_tickers_updates/"
    sectors = save_sp500_sector()
    tickers_df = pd.read_csv(sp500_tickers_location+"sp500.csv")
    tickers = list(tickers_df['Tickers'])
    retrieval_dt = str(datetime.now().date())
    ticker_sector_dic = {tickers[i]: sectors[i] for i in range(len(tickers))}
    print("length of dictionary", len(ticker_sector_dic))
    sector_df = pd.DataFrame(list(ticker_sector_dic.items()), columns=['ticker', 'sector'])
    sector_df.to_csv(sp500_tickers_location + "sp500_sectors.csv", index=False)
    print("SP500 Sectors Updated by {}.".format(retrieval_dt))


    # from dataframe to dict:
    # sector_dict = df.set_index('ticker').T.to_dict('records')[0]