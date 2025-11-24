import requests

def search_yahoo(query):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if 'quotes' in data:
            return [f"{item['symbol']} - {item.get('shortname', item.get('longname', 'Unknown'))}" for item in data['quotes'] if 'symbol' in item]
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

print("Searching for 'Apple'...")
results = search_yahoo("Apple")
print(results[:3])

print("Searching for '0050'...")
results = search_yahoo("0050")
print(results[:3])
