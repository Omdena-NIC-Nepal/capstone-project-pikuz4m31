import csv
import requests

# Function to save climate news to CSV
def save_to_csv(articles, filename="climate_news_nepal.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "URL", "Published At"])  # CSV Header
        
        for article in articles:
            writer.writerow([article['title'], article['url'], article['publishedAt']])
    
    print(f"Data saved to {filename}")

# Modify fetch_climate_news to return articles so they can be saved
def fetch_climate_news(api_key):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'climate change AND Nepal',  # Update query to include both climate change and Nepal
        'apiKey': api_key,  
        'language': 'en',  
        'pageSize': 10,     # Limit to 10 articles
        'sortBy': 'publishedAt'  # Sort by publication date
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        
        if articles:
            valid_articles = []
            for article in articles:
                url = article['url']
                try:
                    # Check if the URL is valid by making a request to it
                    article_response = requests.get(url)
                    article_response.raise_for_status()  # If status is 404 or other errors, it will raise an exception
                    
                    # If URL is valid, add article to the valid_articles list
                    valid_articles.append(article)
                except requests.exceptions.RequestException as e:
                    print(f"Invalid article URL: {url} - Error: {e}")
            
            if valid_articles:
                save_to_csv(valid_articles)
            else:
                print("No valid articles found.")
        else:
            print("No articles found.")
    else:
        print(f"Error: Unable to fetch data (status code: {response.status_code})")

# Replace 'your_api_key' with your actual API key
api_key = '71a3e9a8eec444cb82087f1b78cc6e8a'  # Replace this with your API key from NewsAPI

# Fetch and save the climate news
fetch_climate_news(api_key)
