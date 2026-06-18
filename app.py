from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.compose import TransformedTargetRegressor

from src.data.featuring import centers as sector_centers
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
    max_price_per_sqm = config["data"].get("max_price_per_sqm")
    if max_price_per_sqm is not None:
        interim = interim[
            interim["price_per_sqm"].isna()
            | (interim["price_per_sqm"] <= max_price_per_sqm)
        ]
    interim = interim.reset_index(drop=True)
    processed = processed.reset_index(drop=True)

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
    centers_df = pd.DataFrame(
        [
            {"sector_center": name, "latitude": lat, "longitude": lon}
            for name, (lat, lon) in sector_centers.items()
        ]
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=centers_df["latitude"],
            lon=centers_df["longitude"],
            mode="markers+text",
            text=centers_df["sector_center"],
            textposition="top right",
            marker={
                "size": 11,
                "color": "#c1121f",
                "symbol": "star",
            },
            name="Sector centers",
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.3f}<br>Lon: %{lon:.3f}<extra></extra>",
        )
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def quality_tab(df: pd.DataFrame) -> None:
    quality_columns = ["price", "area", "price_per_sqm", "city", "sector", "rooms", "street"]
    if "link" in df.columns:
        quality_columns.append("link")

    col1, col2 = st.columns(2)
    with col1:
        high = df.nlargest(50, "price_per_sqm")[quality_columns]
        st.subheader("Highest price per sqm")
        st.dataframe(high, use_container_width=True, hide_index=True)
    with col2:
        low = df.nsmallest(50, "price_per_sqm")[quality_columns]
        st.subheader("Lowest price per sqm")
        st.dataframe(low, use_container_width=True, hide_index=True)

    duplicate_columns = ["price", "area", "city", "sector", "street", "rooms"]
    duplicates = df[df.duplicated(subset=duplicate_columns, keep=False)].sort_values(duplicate_columns)
    st.subheader("Potential duplicates")
    duplicate_view_columns = duplicate_columns + ["price_per_sqm"]
    if "link" in df.columns:
        duplicate_view_columns.append("link")
    st.dataframe(duplicates[duplicate_view_columns], use_container_width=True, hide_index=True)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "price_per_sqm",
        "latitude",
        "longitude",
        "distance_to_sector_center",
    }
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]


def prepare_dashboard_training_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_range: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    data = df.dropna(subset=["price_per_sqm"]).copy()
    if label_range is not None:
        data = data[
            data["price_per_sqm"].between(label_range[0], label_range[1], inclusive="both")
        ].copy()
    if data.empty:
        raise ValueError("No rows remain after applying the selected target range.")
    X = data[feature_columns].copy()
    y = data["price_per_sqm"]

    dropped_columns = [
        column for column in X.columns
        if X[column].isna().all()
    ]
    if dropped_columns:
        X = X.drop(columns=dropped_columns)

    if X.empty:
        raise ValueError("No usable feature columns remain after filtering empty columns.")

    X = X.fillna(X.median(numeric_only=True))
    return X, y, list(X.columns), dropped_columns


def build_training_curve(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame | None:
    if not hasattr(model, "staged_predict"):
        return None

    curve_rows = []
    for stage, (train_preds, test_preds) in enumerate(
        zip(model.staged_predict(X_train), model.staged_predict(X_test)),
        start=1,
    ):
        curve_rows.append(
            {
                "stage": stage,
                "dataset": "Train",
                "loss": mean_squared_error(y_train, train_preds),
            }
        )
        curve_rows.append(
            {
                "stage": stage,
                "dataset": "Test",
                "loss": mean_squared_error(y_test, test_preds),
            }
        )

    return pd.DataFrame(curve_rows)


def get_default_gradient_boosting_params() -> dict:
    return {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_samples_leaf": 2,
        "subsample": 0.9,
        "random_state": 42,
    }


def get_default_neural_network_params() -> dict:
    return {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "batch_size": 64,
        "max_iter": 1200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "tol": 1e-4,
        "random_state": 42,
    }


def get_dashboard_models(
    gradient_boosting_params: dict | None = None,
    neural_network_params: dict | None = None,
) -> dict:
    gb_params = gradient_boosting_params or get_default_gradient_boosting_params()
    nn_params = neural_network_params or get_default_neural_network_params()
    return {
        "Gradient Boosting": GradientBoostingRegressor(**gb_params),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
        "Linear Regression": LinearRegression(),
        "Neural Network": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(**nn_params),
            ),
            transformer=StandardScaler(),
        ),
    }


def get_top_important_features(
    df: pd.DataFrame,
    model_name: str,
    available_features: list[str],
    gradient_boosting_params: dict | None = None,
    label_range: tuple[float, float] | None = None,
    top_n: int = 15,
) -> list[str]:
    models = get_dashboard_models(gradient_boosting_params)
    model = models[model_name]
    if not hasattr(model, "feature_importances_"):
        return available_features[:top_n]

    X, y, used_feature_columns, _ = prepare_dashboard_training_data(
        df,
        available_features,
        label_range=label_range,
    )
    model.fit(X, y)
    importance = (
        pd.DataFrame(
            {
                "feature": used_feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return importance["feature"].tolist()


def build_data_volume_curve(
    X: pd.DataFrame,
    y: pd.Series,
    gradient_boosting_params: dict | None = None,
    neural_network_params: dict | None = None,
) -> pd.DataFrame:
    curve_frames = []
    train_size_grid = np.linspace(0.2, 1.0, 5)
    models = get_dashboard_models(gradient_boosting_params, neural_network_params)
    progress_bar = st.progress(0, text="Starting learning curve computation...")
    status = st.empty()

    for index, (model_name, model) in enumerate(models.items(), start=1):
        status.caption(f"Training learning curve for {model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=train_size_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            shuffle=True,
            random_state=42,
            n_jobs=1,
        )

        curve_frames.append(
            pd.DataFrame(
                {
                    "train_size": train_sizes,
                    "model": model_name,
                    "dataset": "Train",
                    "error": -train_scores.mean(axis=1),
                }
            )
        )
        curve_frames.append(
            pd.DataFrame(
                {
                    "train_size": train_sizes,
                    "model": model_name,
                    "dataset": "Test",
                    "error": -test_scores.mean(axis=1),
                }
            )
        )
        progress_bar.progress(
            index / len(models),
            text=f"Completed {index} of {len(models)} models",
        )

    status.empty()
    progress_bar.empty()
    return pd.concat(curve_frames, ignore_index=True)


def train_dashboard_model(
    df: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    gradient_boosting_params: dict | None = None,
    neural_network_params: dict | None = None,
    label_range: tuple[float, float] | None = None,
):
    models = get_dashboard_models(gradient_boosting_params, neural_network_params)

    X, y, used_feature_columns, dropped_columns = prepare_dashboard_training_data(
        df,
        feature_columns,
        label_range=label_range,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    training_curve = build_training_curve(model, X_train, X_test, y_train, y_test)

    results = X_test.copy()
    results["actual"] = y_test
    results["predicted"] = preds
    results["error"] = results["predicted"] - results["actual"]
    results["absolute_error"] = results["error"].abs()

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
        "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
        "Train RMSE": float(np.sqrt(mean_squared_error(y_train, model.predict(X_train)))),
    }
    diagnostics = {
        "used_feature_columns": used_feature_columns,
        "dropped_columns": dropped_columns,
        "training_curve": training_curve,
        "training_rows": len(X_train),
        "validation_rows": len(X_test),
        "modeling_rows": len(X),
        "X_test": X_test,
        "y_test": y_test,
    }
    return model, results, metrics, diagnostics


def build_permutation_importance_table(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": -importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(25)
    )


def model_tab(processed: pd.DataFrame, interim: pd.DataFrame) -> None:
    model_name = st.selectbox(
        "Model",
        ["Gradient Boosting", "Random Forest", "Linear Regression", "Neural Network"],
    )
    available_features = get_model_feature_columns(processed)
    gradient_boosting_params = None
    neural_network_params = None

    label_source = processed["price_per_sqm"].dropna()
    label_min = float(label_source.quantile(0.01)) if len(label_source) else 0.0
    label_max = float(label_source.quantile(0.99)) if len(label_source) else 1.0
    selected_label_range = st.slider(
        "Target range (price per sqm) for training and validation",
        min_value=float(label_source.min()) if len(label_source) else 0.0,
        max_value=float(label_source.max()) if len(label_source) else 1.0,
        value=(label_min, label_max),
    )

    if model_name == "Gradient Boosting":
        with st.expander("Gradient Boosting Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            n_estimators = col1.slider("Estimators", min_value=100, max_value=800, value=400, step=50)
            learning_rate = col2.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
            max_depth = col3.slider("Max depth", min_value=2, max_value=8, value=5, step=1)

            col1, col2 = st.columns(2)
            min_samples_leaf = col1.slider("Min samples leaf", min_value=1, max_value=10, value=2, step=1)
            subsample = col2.slider("Subsample", min_value=0.5, max_value=1.0, value=0.9, step=0.05)

            gradient_boosting_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "subsample": subsample,
                "random_state": 42,
            }
    elif model_name == "Neural Network":
        with st.expander("Neural Network Settings", expanded=True):
            layer_count = st.slider("Hidden layers", min_value=1, max_value=5, value=2, step=1)
            neuron_columns = st.columns(min(layer_count, 3))
            hidden_layer_sizes = []
            for layer_index in range(layer_count):
                column = neuron_columns[layer_index % len(neuron_columns)]
                hidden_layer_sizes.append(
                    column.slider(
                        f"Neurons in layer {layer_index + 1}",
                        min_value=8,
                        max_value=512,
                        value=128 if layer_index == 0 else 64,
                        step=8,
                        key=f"nn_layer_{layer_index + 1}",
                    )
                )

            col1, col2, col3 = st.columns(3)
            activation = col1.selectbox("Activation", ["relu", "tanh", "logistic"], index=0)
            alpha = col2.number_input("L2 regularization (alpha)", min_value=0.0, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")
            learning_rate_init = col3.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")

            col1, col2, col3 = st.columns(3)
            batch_size = col1.slider("Batch size", min_value=16, max_value=256, value=64, step=16)
            max_iter = col2.slider("Max epochs", min_value=100, max_value=3000, value=1200, step=100)
            early_stopping = col3.checkbox("Use early stopping", value=True)

            neural_network_params = {
                "hidden_layer_sizes": tuple(hidden_layer_sizes),
                "activation": activation,
                "alpha": alpha,
                "learning_rate_init": learning_rate_init,
                "batch_size": batch_size,
                "max_iter": max_iter,
                "early_stopping": early_stopping,
                "validation_fraction": 0.15,
                "n_iter_no_change": 30,
                "tol": 1e-4,
                "random_state": 42,
            }

    top_feature_preset = None
    if model_name in {"Gradient Boosting", "Random Forest"}:
        top_feature_preset = get_top_important_features(
            processed,
            model_name,
            available_features,
            gradient_boosting_params=gradient_boosting_params,
            label_range=selected_label_range,
            top_n=15,
        )

    preset = st.radio(
        "Feature preset",
        ["All features", "Top 15 important features", "No geocode-imputed features", "Manual"],
        horizontal=True,
    )
    if preset == "All features":
        default_features = available_features
    elif preset == "Top 15 important features":
        default_features = top_feature_preset or available_features[:15]
    elif preset == "No geocode-imputed features":
        default_features = [
            feature
            for feature in available_features
            if not feature.endswith("_for_model") and feature != "geocode_missing"
        ]
    else:
        default_features = available_features

    feature_columns = st.multiselect(
        "Features",
        available_features,
        default=default_features,
    )
    st.caption(f"Using {len(feature_columns)} of {len(available_features)} available features.")

    if not feature_columns:
        st.warning("Select at least one feature to train a model.")
        return

    try:
        model, results, metrics, diagnostics = train_dashboard_model(
            processed,
            model_name,
            feature_columns,
            gradient_boosting_params=gradient_boosting_params,
            neural_network_params=neural_network_params,
            label_range=selected_label_range,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    if diagnostics["dropped_columns"]:
        st.info(
            "Skipped fully empty features: "
            + ", ".join(diagnostics["dropped_columns"])
        )
    st.caption(
        f"Modeling rows: {diagnostics['modeling_rows']:,} "
        f"(train: {diagnostics['training_rows']:,}, validation: {diagnostics['validation_rows']:,})"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{metrics['MAE']:,.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
    col3.metric("R2", f"{metrics['R2']:.3f}")
    col4.metric("Train MAE", f"{metrics['Train MAE']:,.2f}")
    col5.metric("Train RMSE", f"{metrics['Train RMSE']:,.2f}")

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

    st.subheader("Loss vs Training Stage")
    if diagnostics["training_curve"] is not None:
        fig = px.line(
            diagnostics["training_curve"],
            x="stage",
            y="loss",
            color="dataset",
            markers=True,
            title="Train and test loss across boosting stages",
            labels={"stage": "Training stage", "loss": "Mean squared error"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loss-by-stage chart is available for Gradient Boosting. Random Forest and Linear Regression do not train in epochs/stages.")

    st.subheader("Error vs Training Data Volume")
    if st.button("Build learning curves for all models"):
        X_curve, y_curve, _, _ = prepare_dashboard_training_data(
            processed,
            feature_columns,
            label_range=selected_label_range,
        )
        curves = build_data_volume_curve(
            X_curve,
            y_curve,
            gradient_boosting_params=gradient_boosting_params,
            neural_network_params=neural_network_params,
        )
        fig = px.line(
            curves,
            x="train_size",
            y="error",
            color="model",
            line_dash="dataset",
            markers=True,
            title="Train and test MAE vs training set size",
            labels={"train_size": "Training rows", "error": "MAE"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Builds train/test MAE curves for Gradient Boosting, Random Forest, Linear Regression, and Neural Network.")

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

    st.subheader("Permutation Importance")
    if st.button("Compute permutation importance"):
        permutation_table = build_permutation_importance_table(
            model,
            diagnostics["X_test"],
            diagnostics["y_test"],
        )
        fig = px.bar(
            permutation_table,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="Permutation importance (MAE increase)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(permutation_table, use_container_width=True, hide_index=True)
    else:
        st.caption("Computes permutation importance on the validation split for the current model.")

    st.subheader("Largest errors")
    worst_count = st.slider("Rows to highlight", min_value=10, max_value=200, value=50, step=10)
    worst = results.nlargest(worst_count, "absolute_error").copy()
    detail_columns = [
        column
        for column in ["link", "city", "sector", "street", "price", "area", "rooms"]
        if column in interim.columns
    ]
    detail_columns_to_join = [column for column in detail_columns if column not in worst.columns]
    if detail_columns_to_join:
        worst = worst.join(interim.loc[worst.index, detail_columns_to_join])
    st.dataframe(
        worst[detail_columns + ["actual", "predicted", "error", "absolute_error"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Largest Error Feature Check")
    diagnostic_feature = st.selectbox(
        "Compare price per sqm against feature",
        diagnostics["used_feature_columns"],
        index=diagnostics["used_feature_columns"].index("area") if "area" in diagnostics["used_feature_columns"] else 0,
    )
    diagnostic_data = results.copy()
    diagnostic_data["largest_error"] = "Other test rows"
    diagnostic_data.loc[worst.index, "largest_error"] = "Largest errors"

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            diagnostic_data,
            x=diagnostic_feature,
            y="actual",
            color="largest_error",
            hover_data=["actual", "predicted", "error", "absolute_error"],
            title=f"Actual price per sqm vs {diagnostic_feature}",
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            diagnostic_data,
            x=diagnostic_feature,
            y="absolute_error",
            color="largest_error",
            hover_data=["actual", "predicted", "error", "absolute_error"],
            title=f"Absolute error vs {diagnostic_feature}",
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Largest Error Feature Distribution")
    distribution_feature = st.selectbox(
        "Distribution by feature value",
        diagnostics["used_feature_columns"],
        index=diagnostics["used_feature_columns"].index(diagnostic_feature) if diagnostic_feature in diagnostics["used_feature_columns"] else 0,
    )
    distribution_data = diagnostic_data[[distribution_feature, "largest_error"]].copy()

    unique_values = distribution_data[distribution_feature].nunique(dropna=False)
    if unique_values <= 20:
        counts = (
            distribution_data
            .groupby([distribution_feature, "largest_error"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            counts,
            x=distribution_feature,
            y="count",
            color="largest_error",
            barmode="group",
            title=f"Distribution of largest errors by {distribution_feature}",
        )
    else:
        fig = px.histogram(
            distribution_data,
            x=distribution_feature,
            color="largest_error",
            nbins=40,
            barmode="overlay",
            histnorm="probability density",
            title=f"Distribution of largest errors by {distribution_feature}",
            opacity=0.65,
        )
    st.plotly_chart(fig, use_container_width=True)


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
        model_tab(processed, interim)


if __name__ == "__main__":
    main()
