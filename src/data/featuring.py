import logging
from pathlib import Path
import pandas as pd
import math
import re
import time
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.getLogger("geopy").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

centers = {'Буюканы': (47.039, 28.803),
            'Центр': (47.024, 28.832),
            'Рышкановка': (47.049, 28.863),
            'Старая Почта': (47.061, 28.806), 
            'Ботаника': (46.988, 28.854),
            'Скулянка': (47.038, 28.803),
            'Чокана': (47.036, 28.889),
            'Дурлешты': (47.019, 28.763),
            'Криково': (47.138, 28.862),
            'Ставчены': (47.096, 28.867),
            'Бубуечь': (46.984, 28.931),
            'Телецентр':(47.003, 28.817)
        }

city_names = {
    "Кишинёв": "Chișinău",
    "Дурлешты": "Durlești",
    "Криково": "Cricova",
    "Ставчены": "Stăuceni",
    "Бубуечь": "Bubuieci",
}

sector_names = {
    "Ботаника": "Botanica",
    "Буюканы": "Buiucani",
    "Рышкановка": "Râșcani",
    "Скулянка": "Sculeni",
    "Старая Почта": "Poșta Veche",
    "Телецентр": "Telecentru",
    "Центр": "Centru",
    "Чокана": "Ciocana",
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_center_coordinate(lat, lon) -> bool:
    return any(
        math.isclose(float(lat), center_lat, abs_tol=1e-9)
        and math.isclose(float(lon), center_lon, abs_tol=1e-9)
        for center_lat, center_lon in centers.values()
    )


def normalize_street(street: str | float | None) -> str | None:
    if pd.isna(street):
        return None

    street = str(street).split(";")[0].strip()
    street = re.sub(r"\s+", " ", street)
    street = street.strip(" ,")

    if not street or re.fullmatch(r"[-./*\\]+", street):
        return None

    replacements = [
        (r"^(ул\.?|улица)\s+", "str. "),
        (r"^(пр-т|проспект|пр\.)\s+", "bd. "),
        (r"^(бул\.?|бульвар)\s+", "bd. "),
        (r"^(шос\.?|шоссе)\s+", "șos. "),
        (r"^str\.", "str. "),
        (r"^bd\.", "bd. "),
        (r"^bulevardul\s+", "bd. "),
        (r"^strada\s+", "str. "),
    ]
    for pattern, replacement in replacements:
        street = re.sub(pattern, replacement, street, flags=re.IGNORECASE)

    street = street.replace("Chişinău", "Chișinăului")
    street = street.replace("Nicolai", "Nicolae")
    street = street.replace("Nikolai", "Nicolae")
    street = street.replace("Testimiteanu", "Testemițanu")
    street = street.replace("Miorita", "Miorița")
    street = street.replace("Albisoara", "Albișoara")
    street = street.replace("Durlesti", "Durlești")
    street = street.replace("Алба-Юлия", "Alba-Iulia")
    street = street.replace("Алба Юлия", "Alba-Iulia")
    street = street.replace("Язулуй", "Iazului")
    street = street.replace("Дачия", "Dacia")
    street = street.replace("Штефан Няга", "Ștefan Neaga")
    street = street.replace("Николае Димо", "Nicolae Dimo")

    street = re.sub(r"\s+", " ", street)
    street = street.strip(" ,.-/*\\")
    return street or None


def geocode_query_candidates(row) -> list[str]:
    street = normalize_street(row["street"])
    if not street:
        return []

    city = city_names.get(row["city"], row["city"])
    sector = sector_names.get(row["sector"], row["sector"])

    candidates = [
        f"{street}, {city}, Moldova",
    ]
    if city == "Chișinău" and pd.notna(sector):
        candidates.append(f"{street}, {sector}, Chișinău, Moldova")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def expected_center(row) -> tuple[float, float] | None:
    if row["city"] == "Кишинёв":
        return centers.get(row["sector"])
    return centers.get(row["city"])


def is_plausible_location(row, lat: float, lon: float, max_distance_m: int = 12000) -> bool:
    center = expected_center(row)
    if not center:
        return True
    return haversine(lat, lon, center[0], center[1]) <= max_distance_m


def load_geocode_cache(cache_path: str | Path | None) -> dict:
    if not cache_path or not Path(cache_path).exists():
        return {}

    cache_df = pd.read_csv(cache_path)
    if "status" in cache_df.columns:
        cache_df = cache_df[cache_df["status"].fillna("found") == "found"]

    return {
        row["address"]: (row["latitude"], row["longitude"])
        for _, row in cache_df.dropna(subset=["latitude", "longitude"]).iterrows()
        if not is_center_coordinate(row["latitude"], row["longitude"])
    }


def load_failed_geocode_cache(cache_path: str | Path | None) -> set[str]:
    if not cache_path or not Path(cache_path).exists():
        return set()

    cache_df = pd.read_csv(cache_path)
    if "status" not in cache_df.columns:
        return set()

    failed = cache_df[cache_df["status"].fillna("found").isin(["not_found", "geocode_rejected"])]
    return set(failed["address"].dropna())


def load_unavailable_geocode_cache(cache_path: str | Path | None) -> set[str]:
    if not cache_path or not Path(cache_path).exists():
        return set()

    cache_df = pd.read_csv(cache_path)
    if "status" not in cache_df.columns:
        return set()

    unavailable = cache_df[cache_df["status"].fillna("found") == "geocoder_unavailable"]
    return set(unavailable["address"].dropna())


def save_geocode_cache(
    cache: dict,
    cache_path: str | Path | None,
    failed_addresses: set[str] | None = None,
    unavailable_addresses: set[str] | None = None,
) -> None:
    if not cache_path:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"address": address, "latitude": lat, "longitude": lon, "status": "found"}
        for address, (lat, lon) in sorted(cache.items())
    ]
    if failed_addresses:
        rows.extend(
            {
                "address": address,
                "latitude": None,
                "longitude": None,
                "status": "not_found",
            }
            for address in sorted(failed_addresses)
            if address not in cache
        )
    if unavailable_addresses:
        rows.extend(
            {
                "address": address,
                "latitude": None,
                "longitude": None,
                "status": "geocoder_unavailable",
            }
            for address in sorted(unavailable_addresses)
            if address not in cache and (not failed_addresses or address not in failed_addresses)
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def add_lat_lon(
    data: pd.DataFrame,
    cache_path: str | Path | None = None,
    enable_live_geocoding: bool = False,
    max_geocode_distance_m: int = 12000,
) -> pd.DataFrame:
    cache = load_geocode_cache(cache_path)
    failed_addresses = load_failed_geocode_cache(cache_path)
    unavailable_addresses = load_unavailable_geocode_cache(cache_path)
    geolocator = None
    live_geocoding_available = enable_live_geocoding

    def remember_failure(address: str) -> None:
        failed_addresses.add(address)
        unavailable_addresses.discard(address)
        save_geocode_cache(cache, cache_path, failed_addresses, unavailable_addresses)

    def remember_unavailable(address: str) -> None:
        unavailable_addresses.add(address)
        save_geocode_cache(cache, cache_path, failed_addresses, unavailable_addresses)

    def get_location(row):
        nonlocal geolocator, live_geocoding_available

        if pd.isna(row['street']) or pd.isna(row['city']):
            logging.debug("Missing street or city for row: %s", row.name)
            return None, None
        address = f"{row['street']}, {row['city']}, Moldova"

        if address in cache:
            return cache[address]
        if address in failed_addresses:
            return None, None
        if not live_geocoding_available:
            if enable_live_geocoding:
                remember_unavailable(address)
            return None, None

        if geolocator is None:
            try:
                from geopy.geocoders import Nominatim
            except ModuleNotFoundError:
                logging.debug("Missing geopy; cannot geocode uncached address: %s", address)
                live_geocoding_available = False
                remember_unavailable(address)
                return None, None

            geolocator = Nominatim(user_agent="my_geocoder", timeout=5)

        for query in geocode_query_candidates(row):
            try:
                location = geolocator.geocode(query, limit=1)
                time.sleep(1)
            except Exception as exc:
                logging.warning("Live geocoding unavailable; continuing with cached coordinates only.")
                live_geocoding_available = False
                remember_unavailable(address)
                return None, None
            if location and is_plausible_location(
                row,
                location.latitude,
                location.longitude,
                max_geocode_distance_m,
            ):
                logging.debug(
                    "Found location for %s via %s: %s, %s",
                    address,
                    query,
                    location.latitude,
                    location.longitude,
                )
                cache[address] = (location.latitude, location.longitude)
                failed_addresses.discard(address)
                unavailable_addresses.discard(address)
                save_geocode_cache(cache, cache_path, failed_addresses, unavailable_addresses)
                return location.latitude, location.longitude
            if location:
                logging.debug(
                    "Rejected implausible location for %s via %s: %s, %s",
                    address,
                    query,
                    location.latitude,
                    location.longitude,
                )

        logging.debug("Could not find location for %s", address)
        remember_failure(address)
        return None, None

    locations = [
        get_location(row)
        for _, row in tqdm(
            data.iterrows(),
            total=len(data),
            desc="Geocoding addresses",
            unit="row",
        )
    ]
    data[['latitude', 'longitude']] = pd.DataFrame(locations, index=data.index)
    save_geocode_cache(cache, cache_path, failed_addresses, unavailable_addresses)
    return data


def add_center_distance(data: pd.DataFrame) -> pd.DataFrame:
    def calculate_distance(row):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            logging.debug("Missing latitude or longitude for row: %s", row.name)
            return None
        if row['city'] == 'Кишинёв':
            center_coords = centers.get(row['sector'])
        else:
            center_coords = centers.get(row['city'])
        if not center_coords:
            logging.debug("No center coordinates found for row %s", row.name)
            return None
        return haversine(row['latitude'], row['longitude'], center_coords[0], center_coords[1])

    data['distance_to_sector_center'] = [
        calculate_distance(row)
        for _, row in tqdm(
            data.iterrows(),
            total=len(data),
            desc="Computing center distances",
            unit="row",
        )
    ]
    return data


def add_geo_clusters(data: pd.DataFrame, n_clusters: int = 12) -> pd.DataFrame:
    data = data.copy()
    data["geo_cluster"] = "missing"

    valid_mask = data["latitude"].notna() & data["longitude"].notna()
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        logging.info("Skipping geo clusters because no coordinates are available.")
        return data

    cluster_count = min(n_clusters, valid_count)
    if cluster_count < 1:
        return data

    coordinates = data.loc[valid_mask, ["latitude", "longitude"]]
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coordinates)
    data.loc[valid_mask, "geo_cluster"] = [f"cluster_{label:02d}" for label in labels]
    logging.info("Assigned geo clusters using %s clusters.", cluster_count)
    return data


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    bathrooms_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4 и более': 4}
    geo_columns = ["latitude", "longitude", "distance_to_sector_center"]

    data["geocode_missing"] = data[geo_columns].isna().any(axis=1).astype(int)
    for column in geo_columns:
        data[f"{column}_for_model"] = data[column].fillna(data[column].median())

    data["bathroom_missing"] = data["bathroom"].isna().astype(int)
    bathroom_values = data["bathroom"].astype(str).str.strip().map(bathrooms_mapping)
    data["bathroom"] = bathroom_values.fillna(bathroom_values.median())

    data["apartment_condition_missing"] = data["apartment_condition"].isna().astype(int)
    data["apartment_condition"] = data["apartment_condition"].fillna(data["apartment_condition"].median())

    data["rooms_missing"] = data["rooms"].isna().astype(int)
    data["rooms"] = data["rooms"].fillna(data["rooms"].median())

    data["floor_ratio_missing"] = data["floor_ratio"].isna().astype(int)
    data["floor_ratio"] = data["floor_ratio"].fillna(data["floor_ratio"].median())

    data["geo_cluster_was_missing"] = (data["geo_cluster"] == "missing").astype(int)

    data = pd.get_dummies(data, columns=['city', 'sector', 'geo_cluster'], drop_first=True, dtype=int)
    data = data.drop(columns=["street", "region"])

    numeric_columns = data.select_dtypes(include="number").columns.drop(
        ["price_per_sqm", *geo_columns],
        errors="ignore",
    )
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    return data


def build_features(
    df_cleaned: pd.DataFrame,
    geocode_cache_path: str | Path | None = None,
    enable_live_geocoding: bool = False,
    max_price_per_sqm: int | float | None = None,
    max_geocode_distance_m: int = 12000,
    geo_cluster_count: int = 12,
) -> pd.DataFrame:
    data = df_cleaned.copy()
    rooms_map = {
                    '1-комнатная квартира': 1, 
                    '2-x комнатная квартира': 2, 
                    '3-x комнатная квартира': 3, 
                    '4-x комнатная квартира': 4, 
                    '5-комнатная квартира': 5,
                    '5-комнатная квартира и более': 5,
                    'Комната': 0
                    }
    condition_map = {
                    'Белый вариант': 0,
                    'Нуждается в ремонте': 1,
                    'Косметический pемонт': 2,
                    'Индивидуальный дизайн': 3,
                    'Eвроремонт': 4,
                    }

        
    data["price_per_sqm"] = data["price"] / data["area"]
    if max_price_per_sqm is not None:
        data = data[
            data["price_per_sqm"].isna()
            | (data["price_per_sqm"] <= max_price_per_sqm)
        ]
    data = data.drop(columns=["price"])

    data["rooms"] = data['rooms'].str.extract(r'(\d+)').astype(float)
    data["floor_ratio"] = data["floor"] / data["total_floors"]

    data['has_balcony'] = data['balcony'].notna().astype(int)
    data['has_living'] = data['living'].notna().astype(int)
    data = data.drop(columns=["balcony", "living"])

    data['apartment_condition'] = data['apartment_condition'].map(condition_map)
    data = pd.get_dummies(data, columns=['building_type', 'housing_stock', 'author', 'parcing_space'], drop_first=True, dtype=int)
    data = data.drop(columns=["developer", "layout"]) #too many NaN values

    data = add_lat_lon(
        data,
        geocode_cache_path,
        enable_live_geocoding,
        max_geocode_distance_m,
    )
    data = add_center_distance(data)
    data = add_geo_clusters(data, geo_cluster_count)
    data = prepare_model_features(data)
    return data
    
def run_featuring(config) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    df_cleaned = pd.read_csv(config["data"]["interim_path"])
    data = build_features(
        df_cleaned,
        config["data"].get("geocode_cache_path"),
        config["data"].get("enable_live_geocoding", False),
        config["data"].get("max_price_per_sqm"),
        config["data"].get("max_geocode_distance_m", 12000),
        config["data"].get("geo_cluster_count", 12),
    )
    return data
