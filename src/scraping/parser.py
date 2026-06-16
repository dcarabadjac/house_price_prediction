import logging
from bs4 import BeautifulSoup
import re

def extract_features(soup):
    features = {}

    # находим блок "Характеристики"
    block = soup.find("div", attrs={"data-testid": "Характеристики"})
    if not block:
        return features

    # все строки характеристик
    items = block.find_all("li")

    for item in items:
        key_el = item.find("span", class_=lambda x: x and "key" in x)
        if not key_el:
            continue

        key = key_el.get_text(strip=True)

        # значение может быть в span или a
        value_el = (
            item.find("span", class_=lambda x: x and "value" in x)
            or item.find("a")
        )

        if value_el:
            value = value_el.get_text(strip=True)
            features[key] = value

    return features

def extract_info(soup):

    data = {}

    for p in soup.find_all("p"):
        span = p.find("span")
        if not span:
            continue
        # get full text of <p>
        text = p.get_text(separator=" ", strip=True)

        # split into key and value
        if ":" in text:
            key, _ = text.split(":", 1)
            value = span.get_text(strip=True)
            data[key.strip()] = value

    return data

def parse_listing_card(card):
    """
    Parses a listing card element to extract its title, link, price, and unique id.

    Args:
        card (bs4.element.Tag): BeautifulSoup element representing a single listing card.

    Returns:
        dict or None: A dictionary with keys 'id', 'title', 'price', and 'link' if parsing is successful,
                      or None if an error occurs during parsing.

    Notes:
        - If required elements are missing, empty strings or fallback values are used.
        - Logs a warning and returns None on exception.
    """
    try:
        href = card.get("href", "")
        link = "https://999.md" + href if href else ""

        title_el = (
            card.select_one("h4")
            or card.select_one('[class*="title"]')
            or card.select_one('[class*="content"]')
        )
        title = title_el.get_text(strip=True) if title_el else ""

        price_el = (
            card.select_one(".styles_price__text__VPLPL")
            or card.select_one('[class*="price"]')
        )
        price = price_el.get_text(strip=True) if price_el else ""

        match = re.search(r"/(\d+)(?:\?|$)", href)
        data_id = match.group(1) if match else href.split("?")[0].split("/")[-1] if href else link

        return {
            "id": data_id,
            "title": title,
            "price": price,
            "link": link
        }

    except Exception as e:
        logging.warning(f"Ошибка при разборе карточки: {e}")
        return None

def parse_listing_details(soup):
    """
    Opens the detailed page of a listing and collects all characteristics.

    Args:
        driver: A Selenium WebDriver instance used to navigate web pages.
        link (str): The URL of the listing to be parsed.

    Returns:
        dict: A dictionary containing the listing features if the listing is for sale,
              or None if the listing is not for sale or an error occurs.
    """
    try:
        # Проверяем тип объявления
        data = extract_info(soup)
        ad_type = data.get("Тип предложения", "")
        #print(ad_type)
        #if ad_type != "Продам":
        #    return None  # пропускаем не-продающие объявления

        # Все характеристики <ul>
        features = {}
        ul = soup.select("ul li.styles_group__feature__5ZWJy")
        for li in ul:
            key_el = li.select_one("span.styles_group__key__uRhnQ")
            value_el = li.select_one("span.styles_group__value__XN7OI")
            if key_el and value_el:
                key = key_el.get_text(strip=True)
                value = value_el.get_text(strip=True)
                features[key] = value

        # Добавляем Характеристики из блока "Характеристики"
        features = {**features, **extract_features(soup)}

        address_el = soup.select_one("div.styles_map__title__UgISm")
        address = address_el.get_text(strip=True) if address_el else None
        features["Адрес"] = address

        # Добавляем тип объявления
        features["Тип"] = ad_type

        return features

    except Exception as e:
        logging.warning(f"Ошибка при разборе деталей: {e}")
        return None
