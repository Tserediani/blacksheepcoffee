# Black Sheep Coffee Scraper

## Overview:
This Python script is designed to scrape menu data from the Black Sheep Coffee website. It extracts information about venues, menu categories, menu bundles, menu items, and upsell categories. The scraped data is then processed and saved in text and JSON files.

## Configuration:
- *SCRAPINGBEE_API_KEY:* API key for ScrapingBee (optional).
- *VENUE_LIMIT:* Maximum number of venues to scrape.
- *POOL:* Maximum number of concurrent tasks.
- *DATA_DIRECTORY:* Directory for scraped data.

## ScrapingBee Configuration (Optional):
- If you have a ScrapingBee API key, set `SCRAPINGBEE` to `True` and provide your API key in `SCRAPINGBEE_API_KEY`.

## Running the Script:
- Execute `python blacksheep.py` to start the script.
- Results will be saved in the scripts directory.

## Logging:
- Logs are stored in the "logs" directory with filenames like "blacksheepcoffee_YYYY-MM-DD.log".
- Logs capture script activities, warnings, and errors.

## Output:
- **Menu Output Headers:** ['_date', '_operator', '_brand', '_siteid', '_menu', '_cat', '_subcat', '_code', '_name', '_port', '_stdserve', '_price', '_incdrink', '_stock', '_desc']
- **Venue Output Headers:** ['name', 'address', 'postcode', 'uuid', 'lat', 'long']

## Note:
- Install required dependencies using `pip install -r requirements.txt`.
- Be responsible and comply with the website's terms of service when scraping.
