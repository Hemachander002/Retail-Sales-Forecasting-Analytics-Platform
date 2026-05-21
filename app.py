import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import warnings

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Retail Forecast AI",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #333;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📈 AI Powered Retail Sales Forecasting Platform")

st.markdown("""
Forecast future retail sales using Machine Learning and Time-Series Analytics
""")

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "artifacts/model_trainer/model.joblib"
    )

    return model

model = load_model()

# =========================================================
# LOAD ENCODERS
# =========================================================

@st.cache_resource
def load_encoders():

    encoders = joblib.load(
        "artifacts/data_transformation/label_encoder.joblib"
    )

    return encoders

encoders = load_encoders()

# =========================================================
# LOAD FEATURE COLUMNS
# =========================================================

@st.cache_resource
def load_feature_columns():

    feature_columns = joblib.load(
        "artifacts/model_trainer/features.joblib"
    )

    return feature_columns

feature_columns = load_feature_columns()

# =========================================================
# LOAD HISTORICAL DATA
# =========================================================

@st.cache_data
def load_historical_data():

    historical_data = pd.read_csv(
        "artifacts/data_transformation/combined.csv"
    )

    historical_data["date"] = pd.to_datetime(
        historical_data["date"]
    )

    return historical_data

historical_data = load_historical_data()

# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def preprocess(
    holidays_df,
    oil_df,
    stores_df,
    transactions_df,
    future_data,
    encoders,
    historical_data
):

    # -----------------------------------------------------
    # DATE CONVERSION
    # -----------------------------------------------------

    for df in [
        transactions_df,
        stores_df,
        holidays_df,
        oil_df,
        future_data
    ]:

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"]
            )

    # -----------------------------------------------------
    # MERGING
    # -----------------------------------------------------

    future_data = future_data.merge(
        transactions_df,
        on=["date", "store_nbr"],
        how="left"
    )

    future_data = future_data.merge(
        stores_df,
        on="store_nbr",
        how="left"
    )

    future_data = future_data.merge(
        holidays_df,
        on="date",
        how="left"
    )

    future_data.rename(
        columns={"type_y": "holiday_type"},
        inplace=True
    )

    future_data = future_data.merge(
        oil_df,
        on="date",
        how="left"
    )

    future_data.rename(
        columns={"type_x": "store_type"},
        inplace=True
    )

    # -----------------------------------------------------
    # NULL HANDLING
    # -----------------------------------------------------

    future_data["is_holiday"] = (
        future_data["holiday_type"]
        .notnull()
        .astype(int)
    )

    future_data["holiday_type"] = (
        future_data["holiday_type"]
        .fillna("No Holiday")
    )

    future_data["dcoilwtico"] = (
        future_data["dcoilwtico"]
        .ffill()
        .bfill()
    )

    future_data["transactions"] = (
        future_data["transactions"]
        .fillna(0)
    )

    # -----------------------------------------------------
    # DATE FEATURES
    # -----------------------------------------------------

    future_data["year"] = (
        future_data["date"].dt.year
    )

    future_data["month"] = (
        future_data["date"].dt.month
    )

    future_data["day"] = (
        future_data["date"].dt.day
    )

    future_data["day_of_week"] = (
        future_data["date"].dt.dayofweek
    )

    future_data["week_of_year"] = (
        future_data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    future_data["quarter"] = (
        future_data["date"].dt.quarter
    )

    future_data["is_month_start"] = (
        future_data["date"]
        .dt.is_month_start
        .astype(int)
    )

    future_data["is_month_end"] = (
        future_data["date"]
        .dt.is_month_end
        .astype(int)
    )

    # -----------------------------------------------------
    # DROP UNUSED
    # -----------------------------------------------------

    drop_cols = [
        "locale",
        "locale_name",
        "description",
        "transferred"
    ]

    existing_cols = [
        col for col in drop_cols
        if col in future_data.columns
    ]

    future_data.drop(
        columns=existing_cols,
        inplace=True
    )

    # -----------------------------------------------------
    # FAMILY NAME
    # -----------------------------------------------------

    future_data["family_name"] = (
        future_data["family"]
    )

    # -----------------------------------------------------
    # ENCODING
    # -----------------------------------------------------

    cat_cols = [
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type"
    ]

    for col in cat_cols:

        future_data[col] = future_data[col].apply(
            lambda x:
            encoders[col].transform([x])[0]
            if x in encoders[col].classes_
            else -1
        )

    # -----------------------------------------------------
    # FUTURE FLAG
    # -----------------------------------------------------

    future_data["sales"] = np.nan

    future_data["is_future"] = 1

    historical_data["is_future"] = 0

    # -----------------------------------------------------
    # COMBINE DATA
    # -----------------------------------------------------

    combined_df = pd.concat([
        historical_data,
        future_data
    ])

    # -----------------------------------------------------
    # SORTING
    # -----------------------------------------------------

    combined_df = combined_df.sort_values(
        by=["store_nbr", "family", "date"]
    )

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    combined_df["sales_lag_30"] = (
        combined_df
        .groupby(["store_nbr", "family"])["sales"]
        .shift(30)
    )

    combined_df["rolling_mean_7"] = (
        combined_df
        .groupby(["store_nbr", "family"])["sales_lag_30"]
        .transform(
            lambda x:
            x.rolling(7, min_periods=1).mean()
        )
    )

    combined_df["rolling_std_30"] = (
        combined_df
        .groupby(["store_nbr", "family"])["sales_lag_30"]
        .transform(
            lambda x:
            x.rolling(30, min_periods=1).std()
        )
    )

    # -----------------------------------------------------
    # DROP NaNs
    # -----------------------------------------------------

    combined_df = combined_df[
        combined_df["sales_lag_30"].notnull()
    ]

    # -----------------------------------------------------
    # FUTURE ROWS ONLY
    # -----------------------------------------------------

    future_processed = combined_df[
        combined_df["is_future"] == 1
    ]

    return future_processed

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📂 Upload Files")

future_file = st.sidebar.file_uploader(
    "Upload Future Data CSV",
    type=["csv"]
)

holidays_file = st.sidebar.file_uploader(
    "Upload holidays_events.csv",
    type=["csv"]
)

oil_file = st.sidebar.file_uploader(
    "Upload oil.csv",
    type=["csv"]
)

stores_file = st.sidebar.file_uploader(
    "Upload stores.csv",
    type=["csv"]
)

transactions_file = st.sidebar.file_uploader(
    "Upload transactions.csv",
    type=["csv"]
)

# =========================================================
# MAIN LOGIC
# =========================================================

if (
    future_file
    and holidays_file
    and oil_file
    and stores_file
    and transactions_file
):

    # -----------------------------------------------------
    # LOAD CSVs
    # -----------------------------------------------------

    future_df = pd.read_csv(
        future_file
    )

    holidays_df = pd.read_csv(
        holidays_file
    )

    oil_df = pd.read_csv(
        oil_file
    )

    stores_df = pd.read_csv(
        stores_file
    )

    transactions_df = pd.read_csv(
        transactions_file
    )

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    future_processed = preprocess(
        holidays_df=holidays_df,
        oil_df=oil_df,
        stores_df=stores_df,
        transactions_df=transactions_df,
        future_data=future_df,
        encoders=encoders,
        historical_data=historical_data
    )

    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    X_test = future_processed.drop(
        columns=[
            "sales",
            "date",
            "id",
            "is_future",
            "family_name"
        ],
        errors="ignore"
    )

    # -----------------------------------------------------
    # COLUMN ORDER FIX
    # -----------------------------------------------------

    X_test = X_test[feature_columns]

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    predictions = model.predict(X_test)

    future_processed = future_processed.copy()

    future_processed["predicted_sales"] = (
        predictions
    )

    future_processed["date"] = pd.to_datetime(
        future_processed["date"]
    )

    # =====================================================
# KPI CALCULATIONS
# =====================================================

    daily_sales = (future_processed.groupby("date")["predicted_sales"].sum().reset_index())

    total_sales = daily_sales["predicted_sales"].sum()

    avg_sales = daily_sales["predicted_sales"].mean()

    peak_day = daily_sales.loc[daily_sales["predicted_sales"].idxmax(),"date"]

    top_store = (future_processed.groupby("store_nbr")["predicted_sales"].sum().idxmax())

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Forecasted Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📊 Average Sales",
        f"${avg_sales:,.0f}"
    )

    col3.metric(
        "🏪 Top Store",
        f"Store {top_store}"
    )

    col4.metric(
        "📅 Peak Forecast Day",
        str(peak_day.date())
    )

    st.markdown("---")


    st.sidebar.header("🔍 Filters")

    selected_store = st.sidebar.multiselect(
        "Select Store",
        sorted(
            future_processed["store_nbr"]
            .dropna()
            .unique()
        ),
        default=sorted(
            future_processed["store_nbr"]
            .dropna()
            .unique()
        )
    )

    selected_family = st.sidebar.multiselect(
        "Select Product Family",
        sorted(
            future_processed["family_name"]
            .dropna()
            .unique()
        ),
        default=sorted(
            future_processed["family_name"]
            .dropna()
            .unique()
        )
    )

    filtered_df = future_processed[
        (
            future_processed["store_nbr"]
            .isin(selected_store)
        )
        &
        (
            future_processed["family_name"]
            .isin(selected_family)
        )
    ]


    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Overview",
        "🏪 Store Analytics",
        "🛒 Product Analytics",
        "🤖 Model Intelligence"
    ])


    with tab1:

        st.subheader(
            "Day-by-Day Future Sales Forecast"
        )

        forecast_daily = (
            filtered_df
            .groupby("date")[
                "predicted_sales"
            ]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        forecast_daily["date"] = pd.to_datetime(
            forecast_daily["date"]
        )

        if forecast_daily.empty:

            st.error(
                "No forecast data available."
            )

        else:

            total_future_sales = (
                forecast_daily["predicted_sales"]
                .sum()
            )

            avg_future_sales = (
                forecast_daily["predicted_sales"]
                .mean()
            )

            best_day = (
                forecast_daily.loc[
                    forecast_daily[
                        "predicted_sales"
                    ].idxmax(),
                    "date"
                ]
            )

            highest_sales = (
                forecast_daily[
                    "predicted_sales"
                ].max()
            )

            kpi1, kpi2, kpi3 = st.columns(3)

            kpi1.metric(
                "💰 Total Future Sales",
                f"${total_future_sales:,.0f}"
            )

            kpi2.metric(
                "📊 Average Daily Sales",
                f"${avg_future_sales:,.0f}"
            )

            kpi3.metric(
                "📈 Peak Forecast Day",
                f"{best_day.date()}"
            )

            st.markdown("---")

            forecast_chart = px.bar(

                forecast_daily,

                x="date",

                y="predicted_sales",

                title=(
                    "Predicted Future Sales "
                    "(Day-by-Day)"
                ),

                text_auto=".2s"
            )

            forecast_chart.update_layout(

                template="plotly_dark",

                title_x=0.5,

                xaxis_title="Future Dates",

                yaxis_title="Predicted Sales",

                height=600,

                hovermode="x unified"
            )

            forecast_chart.update_traces(

                hovertemplate=
                "<b>Date:</b> %{x}<br>" +
                "<b>Predicted Sales:</b> %{y:,.2f}<br>",

                textposition="outside"
            )

            st.plotly_chart(
                forecast_chart,
                use_container_width=True
            )

            st.success(
                f"""
                Forecasting engine predicts a peak
                sales value of approximately
                ${highest_sales:,.0f}
                during the future prediction window.
                """
            )

            st.subheader(
                "📋 Daily Forecast Table"
            )

            forecast_daily["predicted_sales"] = (
                forecast_daily[
                    "predicted_sales"
                ].round(2)
            )

            st.dataframe(
                forecast_daily,
                use_container_width=True
            )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.subheader(
            "Store-wise Forecast"
        )

        store_sales = (
            filtered_df
            .groupby("store_nbr")[
                "predicted_sales"
            ]
            .sum()
            .reset_index()
        )

        store_chart = px.bar(
            store_sales,
            x="store_nbr",
            y="predicted_sales",
            title="Forecasted Sales by Store"
        )

        st.plotly_chart(
            store_chart,
            use_container_width=True
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.subheader(
            "Product Family Forecast"
        )

        family_sales = (
            filtered_df
            .groupby("family_name")[
                "predicted_sales"
            ]
            .sum()
            .reset_index()
        )

        family_chart = px.bar(
            family_sales,
            x="predicted_sales",
            y="family_name",
            orientation="h",
            title="Forecasted Sales by Product Family"
        )

        st.plotly_chart(
            family_chart,
            use_container_width=True
        )

    # =====================================================
    # TAB 4
    # =====================================================

    with tab4:

        st.subheader(
            "Feature Importance"
        )

        feature_importance = pd.DataFrame({

            "Feature": X_test.columns,

            "Importance":
            model.feature_importances_

        })

        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(15)
        )

        importance_chart = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Important Features"
        )

        st.plotly_chart(
            importance_chart,
            use_container_width=True
        )

    # =====================================================
    # DOWNLOADS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📥 Download Forecasts"
    )

    csv = future_processed.to_csv(
        index=False
    )

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

else:

    st.info(
        """
        Upload all required CSV files
        to generate future sales forecasts.
        """
    )