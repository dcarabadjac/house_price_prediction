import logging
from pathlib import Path
import pandas as pd
import math

centers = {'Буюканы': (47.039, 28.803),
            'Центр': (47.024, 28.832),
            'Рышкановка': (47.049, 28.863),
            'Старая Почта': (47.061, 28.806), 
            'Ботаника': (46.988, 28.854),
            'Скулянка': (47.038, 28.803),
            'Чокана': (47.036, 28.889),
            'Дурлешты': (47.019, 28.763),
            'Крикова': (47.138, 28.862),
            'Ставчены': (47.096, 28.867),
            'Бубуечь': (46.984, 28.931),
            'Телецентр':(47.003, 28.817)
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


def load_geocode_cache(cache_path: str | Path | None) -> dict:
    if not cache_path or not Path(cache_path).exists():
        return {}

    cache_df = pd.read_csv(cache_path)
    return {
        row["address"]: (row["latitude"], row["longitude"])
        for _, row in cache_df.dropna(subset=["latitude", "longitude"]).iterrows()
    }


def save_geocode_cache(cache: dict, cache_path: str | Path | None) -> None:
    if not cache_path:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"address": address, "latitude": lat, "longitude": lon}
        for address, (lat, lon) in sorted(cache.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def add_lat_lon(data: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
    cache = load_geocode_cache(cache_path)
    geocode = None

    def get_location(row):
        nonlocal geocode

        if pd.isna(row['street']) or pd.isna(row['city']):
            print(f"Missing street or city for row: {row.name}")
            return None, None
        address = f"{row['street']}, {row['city']}, Moldova"

        if address in cache:
            return cache[address]

        if geocode is None:
            try:
                from geopy.extra.rate_limiter import RateLimiter
                from geopy.geocoders import Nominatim
            except ModuleNotFoundError as exc:
                if row['city'] == 'Кишинёв':
                    center_coords = centers.get(row['sector'])
                else:
                    center_coords = centers.get(row['city'])

                if center_coords:
                    print(f"Using center coordinates for uncached address: {address}")
                    cache[address] = center_coords
                    return center_coords

                print(f"Missing geopy and no center coordinates for uncached address: {address}")
                return None, None

            geolocator = Nominatim(user_agent="my_geocoder")
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        location = geocode(address)
        if location:
            print(f"Found location for {address}: {location.latitude}, {location.longitude}")
            cache[address] = (location.latitude, location.longitude)
            return location.latitude, location.longitude
        print(f"Could not find location for {address}") 
        return None, None

    data[['latitude', 'longitude']] = data.apply(
        lambda row: pd.Series(get_location(row)), axis=1
    )
    save_geocode_cache(cache, cache_path)
    return data


def add_center_distance(data: pd.DataFrame) -> pd.DataFrame:
    def calculate_distance(row):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            print(f"Missing latitude or longitude for row: {row.name}")
            return None
        if row['city'] == 'Кишинёв':
            center_coords = centers.get(row['sector'])
        else:
            center_coords = centers.get(row['city'])
        if not center_coords:
            print(f"No center coordinates found for sector: {row['sector']}")
            return None
        distance = haversine(row['latitude'], row['longitude'], center_coords[0], center_coords[1])
        print(f"Calculated distance for row {row.name}: {distance} meters")
        return distance

    data['distance_to_sector_center'] = data.apply(calculate_distance, axis=1)
    return data

def build_features(df_cleaned: pd.DataFrame, geocode_cache_path: str | Path | None = None) -> pd.DataFrame:
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
    data = data.drop(columns=["price"])

    data["rooms"] = data['rooms'].str.extract(r'(\d+)').astype(float)
    data["floor_ratio"] = data["floor"] / data["total_floors"]

    data['has_balcony'] = data['balcony'].notna().astype(int)
    data['has_living'] = data['living'].notna().astype(int)
    data = data.drop(columns=["balcony", "living"])

    data['apartment_condition'] = data['apartment_condition'].map(condition_map)
    data = pd.get_dummies(data, columns=['building_type', 'housing_stock', 'author', 'parcing_space'], drop_first=True, dtype=int)
    data = data.drop(columns=["developer", "layout"]) #too many NaN values

    data = add_lat_lon(data, geocode_cache_path)
    data = add_center_distance(data)
    return data
    
def run_featuring(config) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    df_cleaned = pd.read_csv(config["data"]["interim_path"])
    data = build_features(df_cleaned, config["data"].get("geocode_cache_path"))
    return data
