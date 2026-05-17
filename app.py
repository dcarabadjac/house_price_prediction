from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils.config import load_config


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@st.cache_data
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(CONFIG_PATH)
    interim = pd.read_csv(ROOT / config["data"]["interim_path"])
    processed = pd.read_csv(ROOT / config["data"]["processed_path"])

    interim = interim.copy()
    interim["price_per_sqm"] = interim["price"] / interim["area"]

    geo_columns = ["latitude", "longitude", "distance_to_sector_center", "geocode_missing"]
    for column in geo_columns:
        if column in processed.columns:
            interim[column] = processed[column]

    return interim, processed


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar:
        st.header("Filters")

        city_options = sorted(filtered["city"].dropna().unique())
        cities = st.multiselect("City", city_options, default=city_options)
        filtered = filtered[filtered["city"].isin(cities)]

        sector_options = sorted(filtered["sector"].dropna().unique())
        sectors = st.multiselect("Sector", sector_options, default=sector_options)
        filtered = filtered[filtered["sector"].isin(sectors)]

        rooms_options = sorted(filtered["rooms"].dropna().unique())
        rooms = st.multiselect("Rooms", rooms_options, default=rooms_options)
        filtered = filtered[filtered["rooms"].isin(rooms)]

        price_min = int(filtered["price_per_sqm"].quantile(0.01)) if len(filtered) else 0
        price_max = int(filtered["price_per_sqm"].quantile(0.99)) if len(filtered) else 1
        price_range = st.slider(
            "Price per sqm",
            min_value=0,
            max_value=max(price_max, 1),
            value=(max(price_min, 0), max(price_max, 1)),
        )
        filtered = filtered[
            filtered["price_per_sqm"].between(price_range[0], price_range[1], inclusive="both")
        ]

        area_min = int(filtered["area"].min()) if len(filtered) else 0
        area_max = int(filtered["area"].max()) if len(filtered) else 1
        area_range = st.slider(
            "Area",
            min_value=max(area_min, 0),
            max_value=max(area_max, 1),
            value=(max(area_min, 0), max(area_max, 1)),
        )
        filtered = filtered[filtered["area"].between(area_range[0], area_range[1], inclusive="both")]

    return filtered


def metric_row(df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Median €/sqm", f"{df['price_per_sqm'].median():,.0f}")
    col3.metric("Median price", f"{df['price'].median():,.0f}")
    col4.metric("Median area", f"{df['area'].median():,.0f}")
    col5.metric("Geocode missing", f"{int(df.get('geocode_missing', pd.Series(0)).sum()):,}")


def overview_tab(df: pd.DataFrame) -> None:
    metric_row(df)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="price_per_sqm", nbins=60, title="Price per sqm")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="area", nbins=50, title="Area")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        city_summary = (
            df.groupby("city", as_index=False)
            .agg(rows=("price_per_sqm", "size"), median_price_per_sqm=("price_per_sqm", "median"))
            .sort_values("median_price_per_sqm", ascending=False)
        )
        st.dataframe(city_summary, use_container_width=True, hide_index=True)
    with col2:
        missing = (
            df.isna()
            .sum()
            .rename("missing")
            .reset_index()
            .rename(columns={"index": "column"})
            .sort_values("missing", ascending=False)
        )
        st.dataframe(missing, use_container_width=True, hide_index=True)


def explorer_tab(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="sector", y="price_per_sqm", color="city", title="Price per sqm by sector")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            df,
            x="area",
            y="price",
            color="city",
            hover_data=["sector", "rooms", "street"],
            title="Price vs area",
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            df,
            x="distance_to_sector_center",
            y="price_per_sqm",
            color="sector",
            hover_data=["city", "rooms", "street"],
            title="Distance to center vs price per sqm",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="rooms", y="price_per_sqm", color="city", title="Price per sqm by rooms")
        st.plotly_chart(fig, use_container_width=True)


def map_tab(df: pd.DataFrame) -> None:
    mapped = df.dropna(subset=["latitude", "longitude"]).copy()
    if mapped.empty:
        st.info("No coordinates available.")
        return

    fig = px.scatter_mapbox(
        mapped,
        lat="latitude",
        lon="longitude",
        color="price_per_sqm",
        size="area",
        hover_data=["city", "sector", "rooms", "price", "price_per_sqm", "street", "geocode_missing"],
        color_continuous_scale="Viridis",
        zoom=10,
        height=720,
        title="Listings map",
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def quality_tab(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        high = df.nlargest(50, "price_per_sqm")[
            ["price", "area", "price_per_sqm", "city", "sector", "rooms", "street"]
        ]
        st.subheader("Highest price per sqm")
        st.dataframe(high, use_container_width=True, hide_index=True)
    with col2:
        low = df.nsmallest(50, "price_per_sqm")[
            ["price", "area", "price_per_sqm", "city", "sector", "rooms", "street"]
        ]
        st.subheader("Lowest price per sqm")
        st.dataframe(low, use_container_width=True, hide_index=True)

    duplicate_columns = ["price", "area", "city", "sector", "street", "rooms"]
    duplicates = df[df.duplicated(subset=duplicate_columns, keep=False)].sort_values(duplicate_columns)
    st.subheader("Potential duplicates")
    st.dataframe(duplicates[duplicate_columns + ["price_per_sqm"]], use_container_width=True, hide_index=True)


def train_dashboard_model(df: pd.DataFrame, model_name: str):
    models = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    data = df.dropna(subset=["price_per_sqm"])
    X = data.drop(columns=["price_per_sqm", "latitude", "longitude", "distance_to_sector_center"])
    y = data["price_per_sqm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = X_test.copy()
    results["actual"] = y_test
    results["predicted"] = preds
    results["error"] = results["predicted"] - results["actual"]
    results["absolute_error"] = results["error"].abs()

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds),
    }
    return model, results, metrics


def model_tab(processed: pd.DataFrame) -> None:
    model_name = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "Linear Regression"])
    model, results, metrics = train_dashboard_model(processed, model_name)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:,.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
    col3.metric("R2", f"{metrics['R2']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(results, x="actual", y="predicted", title="Predicted vs actual")
        fig.add_shape(
            type="line",
            x0=results["actual"].min(),
            y0=results["actual"].min(),
            x1=results["actual"].max(),
            y1=results["actual"].max(),
            line={"dash": "dash", "color": "gray"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(results, x="error", nbins=60, title="Residuals")
        st.plotly_chart(fig, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame({"feature": results.drop(columns=["actual", "predicted", "error", "absolute_error"]).columns,
                          "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(25)
        )
        fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Feature importance")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Largest errors")
    st.dataframe(
        results.nlargest(50, "absolute_error")[["actual", "predicted", "error", "absolute_error"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="House Price Prediction", layout="wide")
    st.title("House Price Prediction")

    interim, processed = load_datasets()
    filtered = filter_data(interim)

    overview, explorer, map_view, quality, model = st.tabs(
        ["Overview", "Explorer", "Map", "Data Quality", "Model Diagnostics"]
    )
    with overview:
        overview_tab(filtered)
    with explorer:
        explorer_tab(filtered)
    with map_view:
        map_tab(filtered)
    with quality:
        quality_tab(filtered)
    with model:
        model_tab(processed)


if __name__ == "__main__":
    main()
