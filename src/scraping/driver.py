import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

try:
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    ChromeDriverManager = None

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-features=Translate,BackForwardCache")
    options.add_argument("--window-size=1440,2200")
    options.page_load_strategy = "eager"
    options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
        },
    )
    options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    try:
        # Prefer Selenium Manager first. It is usually more reliable than a
        # cached webdriver_manager binary when Chrome updates locally.
        driver = webdriver.Chrome(options=options)
    except Exception as selenium_manager_error:
        if ChromeDriverManager is None:
            raise

        logging.warning(
            "Selenium Manager не смог поднять ChromeDriver: %s. Пробую webdriver_manager.",
            selenium_manager_error,
        )
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )

    return driver
