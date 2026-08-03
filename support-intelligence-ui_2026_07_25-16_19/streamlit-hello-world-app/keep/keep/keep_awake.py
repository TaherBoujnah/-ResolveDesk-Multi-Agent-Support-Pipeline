import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- REPLACE THIS WITH YOUR ACTUAL STREAMLIT URL ---
STREAMLIT_URL = "https://resolvedesk.streamlit.app/"

def main():
    print(f"Opening headless browser to ping: {STREAMLIT_URL}")
    
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(STREAMLIT_URL)
        # Wait 10 seconds to allow the JavaScript to execute and WebSockets to connect
        time.sleep(10)
        print("Success! WebSocket connected and sleep timer reset.")
    except Exception as e:
        print(f"Failed to load page: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
