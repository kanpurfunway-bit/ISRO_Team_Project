import os
import sys
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SeedDatabase")

# Add current directory to path to import app.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app import fetch_and_store_weather, INDIAN_CITIES, init_db_pool
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    logger.error(f"Failed to import from app.py: {e}")
    sys.exit(1)

def main():
    if not os.environ.get("DATABASE_URL"):
        logger.error("DATABASE_URL is not set in .env")
        return
    if not os.environ.get("WEATHERAPI_KEY"):
        logger.error("WEATHERAPI_KEY is not set in .env")
        return
        
    logger.info("Initializing database pool...")
    init_db_pool()
    
    end = datetime.now() - timedelta(days=1)
    start = datetime.now() - timedelta(days=31)
    
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching historical weather data from {start_str} to {end_str} for {len(INDIAN_CITIES)} cities...")
    logger.info("This may take several minutes due to API rate limits.")
    
    try:
        count = fetch_and_store_weather(INDIAN_CITIES, start_str, end_str, "history")
        logger.info(f"SUCCESS: Inserted {count} historical records into the database.")
        logger.info("You can now safely hit the /api/train endpoint to train your AI models!")
    except Exception as e:
        logger.error(f"An error occurred while seeding the database: {e}")

if __name__ == "__main__":
    main()
