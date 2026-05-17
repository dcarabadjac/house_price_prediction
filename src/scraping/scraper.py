import logging
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from src.utils.io import save_items_to_csv
from src.scraping.driver import create_driver
from src.scraping.parser import parse_listing_card, parse_listing_details

class Scraper:
    def __init__(self, driver, base_url):
        self.driver = driver
        self.base_url = base_url

    def scrape_page(self, page: int, delay: int):
        url = f"{self.base_url}?page={page}&o_16_1=776"
        logging.info(f"Открываю страницу {page}: {url}")
    
        self.driver.get(url)
        time.sleep(delay)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        cards = soup.select("a.styles_advert__photo__link__SnL_t")
        print(f"Найдено {len(cards)} объявлений на странице {page}")
        return [parse_listing_card(c) for c in cards]

    def scrape_listing(self, link: str):
        self.driver.get(link)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        html = self.driver.page_source
        return parse_listing_details(soup)
    

def run_scraper(config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    
    base_url = config["scraping"]["base_url"]
    first_page = config["scraping"]["first_page"]
    n_pages = config["scraping"]["pages"]

    seen_ids = set()
    #driver = create_driver()

    try:
        for page in range(first_page, n_pages + 1):
            detailed_items = []
            driver = create_driver()
            scraper = Scraper(driver, base_url)
            items = scraper.scrape_page(page, config["scraping"]["delay"])
            if not items:
                logging.info("Нет новых объявлений на странице — прекращаю.")
                continue
            for idx, item in tqdm(enumerate(items), total=len(items), desc="Processing items"):

                details = scraper.scrape_listing(item.get("link"))  

                if not details:
                    print('Skipping item due to missing details')
                    continue

                item.update(details)
                detailed_items.append(item)
            save_items_to_csv(detailed_items, config["data"]["raw_prefix"] + f"_page_{page}.csv")

    finally:
        driver.quit()
