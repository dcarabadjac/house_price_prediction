import logging
from pathlib import Path
import re
from typing import Iterable
import pandas as pd

allowed_cities = ['Кишинёв',
            'Дурлешты',
            'Криково',
            'Ставчены',
            'Бубуечь'
]       

def load_raw_data(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved_paths = []
    for path in paths:
        path = Path(path)
        if any(char in str(path) for char in "*?[]"):
            resolved_paths.extend(sorted(Path().glob(str(path))))
        else:
            resolved_paths.append(path)

    if not resolved_paths:
        raise FileNotFoundError("No raw data files matched the configured paths.")

    dfs = [pd.read_csv(path) for path in resolved_paths]
    return pd.concat(dfs, ignore_index=True)

def drop_bad_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["Высота потолков", "Жилая площадь", "Тип", 
                    "Площадь кухни", "id", "title", "Общая площадь",
                    "Количество этажей", "id", "title", "Общая площадь",
                    "Этаж", "Адрес"]
    return df.drop(columns=cols_to_drop)

def filtering(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    sel_region = df_cleaned["region"] == "Кишинёв мун."
    sel_city = df_cleaned["city"].isin(allowed_cities)
    df_cleaned = df_cleaned[sel_region & sel_city]
    return df_cleaned

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Количество комнат": "rooms",
        "Парковочное место": "parcing_space",
        "Планировка": "layout",
        "Балкон / лоджия": "balcony",
        "Санузел": "bathroom",
        "Ливинг": "living",
        "Автор объявления": "author",
        "Состояние квартиры": "apartment_condition",
        "Тип здания": "building_type",
        "Жилой фонд": "housing_stock",
        "Застройщик": "developer"

    })
    return df
def clean_price(price: str | float | None) -> int | None:
    if pd.isna(price):
        return None

    price = str(price)
    price_match = re.search(r"(\d[\d\s\xa0]*)\s*€", price)
    if not price_match:
        return None

    digits = re.sub(r"\D", "", price_match.group(1))
    return int(digits) if digits else None

def clean_total_floors(total_floors: str | int | None) -> int | None:
    if pd.isna(total_floors):
        return None

    match = re.search(r"\d+", str(total_floors))
    return int(match.group()) if match else None
    
def clean_floor(floor: str | int | None) -> int | None:
    if pd.isna(floor):
        return None

    match = re.search(r"\d+", str(floor))
    return int(match.group()) if match else None

def clean_address(address: str | None) -> tuple | None:
    if pd.isna(address):
        return None, None, None, None
    address = str(address)
    address_info = address.split(", ")
   
    if len(address_info) < 4:
        return address, None, None, None
    region = address_info[0]
    city = address_info[1]
    sector = address_info[2]
    street = address_info[3] + ' ' + address_info[4] if len(address_info) > 4 else address_info[3]
    return region, city, sector, street

def clean_area(area: str | None) -> int | None:
    if pd.isna(area):
        return None

    digits = re.sub(r"\D", "", str(area))
    return int(digits) if digits else None

def drop_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return df

    cleaned_ids = df["id"].astype(str).str.strip()
    valid_ids = cleaned_ids.ne("") & cleaned_ids.ne("nan")
    duplicate_mask = pd.Series(False, index=df.index)
    duplicate_mask.loc[valid_ids] = cleaned_ids.loc[valid_ids].duplicated()
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        logging.info("Удаляю %s дубликатов по id", duplicate_count)

    deduplicated = df.loc[~duplicate_mask]
    return deduplicated.reset_index(drop=True)

def drop_potential_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_columns = ["price", "Общая площадь", "Адрес", "Количество комнат"]
    available_columns = [column for column in duplicate_columns if column in df.columns]
    if len(available_columns) < len(duplicate_columns):
        return df

    duplicate_mask = df.duplicated(subset=duplicate_columns, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        logging.info("Удаляю %s потенциальных дублей по признакам объявления", duplicate_count)

    return df.loc[~duplicate_mask].reset_index(drop=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    if "Тип" in df_cleaned:
        df_cleaned = df_cleaned[df_cleaned["Тип"] == "Продам"]

    df_cleaned = drop_duplicate_ids(df_cleaned)
    df_cleaned = drop_potential_duplicates(df_cleaned)

    df_cleaned["total_floors"] = df_cleaned["Количество этажей"].apply(clean_total_floors)
    df_cleaned["floor"] = df_cleaned["Этаж"].apply(clean_floor)

    df_cleaned["price"] = df_cleaned["price"].apply(clean_price)
    df_cleaned[["region", "city", "sector", "street"]] = df_cleaned["Адрес"].apply(clean_address).apply(pd.Series)
    df_cleaned["area"] = df_cleaned["Общая площадь"].apply(clean_area)
    df_cleaned = rename_columns(df_cleaned)
    df_cleaned = drop_bad_columns(df_cleaned)
    return df_cleaned

def run_cleaning(config) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    raw_paths = config["data"].get("raw_paths", config["data"]["raw_path"])
    raw_data = load_raw_data(raw_paths)
    clean = clean_data(raw_data)
    clean = filtering(clean)
    return clean
