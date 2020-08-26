import requests
import bs4 as bs
import pandas as pd
from datetime import datetime


# Running the code below need to turn on the VPN
def save_sp500_tickers():
    """
    This is a function to save current stock ticker in S&P500

    Input: None

    Output:
    tickers: A list of ticker name of S&P500 stocks
    """
    #     print("Starting to request...")
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #     print("Got the requested website")
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    ### Collect the Ticker Name from Wikipedia tables
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        #         print("Getting Ticker: ", ticker)
        tickers.append(ticker)
    ### Remove the newline character from list
    sample_list = tickers
    converted_list = []
    for element in sample_list:
        converted_list.append(element.strip())
    ### In Yahoo Finance, the website uses '-' in ticker name instead of '.'
    ### Replace "." With "-"
    new_strings = []
    for string in converted_list:
        new_string = string.replace(".", "-")
        new_strings.append(new_string)
    tickers = new_strings

    """
    ### Save the Ticker into file
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)    
    """
    return tickers



if __name__ == "__main__":
    print("Updating SP500 tickers...")
    sp500_tickers_location = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/quant_research/sp500_tickers_updates/"
    sp500_ = save_sp500_tickers()
    retrieval_dt = str(datetime.now().date())
    sp500_df = pd.DataFrame(sp500_, columns=['Tickers'])
    sp500_df['retrieval_dt'] = retrieval_dt
    sp500_df.to_csv(sp500_tickers_location+"sp500.csv",index=False)
    print("SP500 Tickers Updated by {}.".format(retrieval_dt))