import requests

# s = requests.session()
#
# s.keep_alive = False

# response = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

response = requests.get("http://www.baidu.com/")
print(response.content)