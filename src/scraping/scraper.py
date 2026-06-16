import logging
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from src.utils.io import save_items_to_csv
from src.scraping.driver import create_driver
from src.scraping.parser import parse_listing_card, parse_listing_details

class Scraper:
    def __init__(self, driver, base_url):
        self.driver = driver
        self.base_url = base_url

    def restart_driver(self):
        try:
            self.driver.quit()
        except Exception:
            pass
        self.driver = create_driver()
        logging.info("WebDriver перезапущен после сбоя вкладки.")

    def _load_soup(
        self,
        url: str,
        delay: float,
        wait_selector: str | None = None,
        wait_timeout: int | None = None,
    ) -> BeautifulSoup:
        self.driver.get(url)
        if wait_selector:
            WebDriverWait(self.driver, wait_timeout or max(int(delay), 10)).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
        if delay > 0:
            time.sleep(delay)
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def _extract_listing_cards(self, soup: BeautifulSoup):
        selectors = [
            'a.styles_advert__photo__link__SnL_t',
            '[data-testid="infinite-ads-list"] a[href^="/ru/"]',
            'a[href^="/ru/"][href*="/"]',
        ]

        seen_links = set()
        parsed_cards = []
        for selector in selectors:
            cards = soup.select(selector)
            if not cards:
                continue

            for card in cards:
                parsed = parse_listing_card(card)
                if not parsed:
                    continue
                link = parsed.get("link", "")
                if not link or link in seen_links:
                    continue
                if "/booster/" in link or "/add" in link:
                    continue
                if not parsed.get("id"):
                    continue
                seen_links.add(link)
                parsed_cards.append(parsed)

            if parsed_cards:
                logging.info("Карточки найдены через селектор: %s", selector)
                return parsed_cards

        return []

    def scrape_page(self, page: int, delay: int):
        url = f"{self.base_url}?page={page}&o_16_1=776"
        logging.info(f"Открываю страницу {page}: {url}")

        try:
            soup = self._load_soup(
                url,
                delay,
                wait_selector='[data-testid="infinite-ads-list"]',
                wait_timeout=max(delay, 10),
            )
        except TimeoutException:
            logging.warning("Контейнер объявлений не появился вовремя на странице %s", page)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
        except WebDriverException as exc:
            logging.warning("Ошибка браузера на странице %s: %s", page, exc)
            self.restart_driver()
            soup = self._load_soup(
                url,
                delay,
                wait_selector='[data-testid="infinite-ads-list"]',
            )

        items = self._extract_listing_cards(soup)
        print(f"Найдено {len(items)} объявлений на странице {page}")
        return items

    def scrape_listing(self, link: str, delay: float, retries: int = 2):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                soup = self._load_soup(link, delay, wait_timeout=8)
                return parse_listing_details(soup)
            except WebDriverException as exc:
                last_error = exc
                logging.warning(
                    "Сбой браузера при открытии %s (попытка %s/%s): %s",
                    link,
                    attempt,
                    retries,
                    exc,
                )
                self.restart_driver()

        logging.warning("Не удалось открыть объявление после %s попыток: %s", retries, link)
        if last_error:
            logging.debug("Последняя ошибка WebDriver: %s", last_error)
        return None
    

def run_scraper(config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    
    base_url = config["scraping"]["base_url"]
    first_page = config["scraping"]["first_page"]
    n_pages = config["scraping"]["pages"]
    list_delay = config["scraping"]["delay"]
    detail_delay = config["scraping"].get("detail_delay", 0.6)

    driver = None
    try:
        driver = create_driver()
        scraper = Scraper(driver, base_url)
        for page in range(first_page, n_pages + 1):
            detailed_items = []
            items = scraper.scrape_page(page, list_delay)
            if not items:
                logging.warning("На странице %s объявления не распознаны. Пробую следующую страницу.", page)
                continue
            for idx, item in tqdm(enumerate(items), total=len(items), desc="Processing items"):
                details = scraper.scrape_listing(item.get("link"), detail_delay)

                if not details:
                    print('Skipping item due to missing details')
                    continue

                item.update(details)
                detailed_items.append(item)
            save_items_to_csv(detailed_items, config["data"]["raw_prefix"] + f"_page_{page}.csv")
    finally:
        if driver is not None:
            driver.quit()
