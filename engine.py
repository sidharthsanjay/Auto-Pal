import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px

def render_engine_dashboard():
    # st.set_page_config(page_title="Engine Dashboard", layout="wide")

    # ---------- MongoDB Connection ----------
    @st.cache_resource
    def connect_to_mongo():
        mongo_url = " "  # your MongoDB connection string
        client = MongoClient(mongo_url)
        db = client[" "] # your database name
        collection = db[" "] # your collection name
        return collection

    collection = connect_to_mongo()

    # ---------- Load Data ----------
    @st.cache_data(ttl=60)
    def load_data():
        data = list(collection.find())
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip()
        return df

    df = load_data()

    st.markdown("<h1 style='text-align:center;'>Engine Dashboard</h1>", unsafe_allow_html=True)

    # ---------- Valid Source Status Fields ----------
    status_sources = [
        "onthemarket_let", "onthemarket_sale",
        "rightmove_commercial", "rightmove_let", "rightmove_sale",
        "zoopla_commercial", "zoopla_let", "zoopla_sale"
    ]

    # ---------- UI Filter: Select Source for Chart ----------
    col1 = st.columns(1)[0]
    with col1:
        selected_source = st.selectbox("Select Source (for Chart & Table)", status_sources)

    # ---------- Pie Chart and Summary Table ----------
    st.subheader(f"Status Overview for {selected_source}")
    status_column = f"{selected_source}_status"
    updated_date_column = f"{selected_source}_updated_datetime"

    if status_column in df.columns and updated_date_column in df.columns:
        pie_df = df.copy()

        pie_df[status_column] = pie_df[status_column].fillna("").astype(str).str.strip().str.lower()
        pie_df[status_column] = pie_df[status_column].replace("", "null")

        counts = {
            "Completed": pie_df[status_column].str.contains("completed").sum(),
            "Failed": pie_df[status_column].str.contains("failed").sum(),
            "Null": (pie_df[status_column] == "null").sum()
        }

        total = sum(counts.values())
        if total > 0:
            percentages = {k: f"{(v/total)*100:.2f}%" for k, v in counts.items()}
            summary_df = pd.DataFrame({
                "Status": ["Completed", "Failed", "Null"],
                "Count": [counts["Completed"], counts["Failed"], counts["Null"]],
                "Percentage": [percentages["Completed"], percentages["Failed"], percentages["Null"]]
            })

            update_df = df.copy()
            update_df[updated_date_column] = pd.to_datetime(update_df[updated_date_column], errors='coerce')
            update_df = update_df.dropna(subset=[updated_date_column])
    
            update_df["updated_date_str"] = update_df[updated_date_column].dt.date.astype(str)
            updated_count_df = update_df["updated_date_str"].value_counts().reset_index()
            updated_count_df.columns = ["Date", "Count"]

            chart_col1, chart_col2, table_col = st.columns([1,1,1])
            with chart_col1:
                st.markdown("### Update Date")
                fig1 = px.pie(
                    updated_count_df,
                    names="Date",
                    values="Count",
                    hole=0.3
                )
                fig1.update_traces(textinfo="none")  # Hide labels
                st.plotly_chart(fig1, use_container_width=True)

            with chart_col2:
                st.markdown("### Status Distribution")
                fig2 = px.pie(
                    summary_df,
                    names="Status",
                    values="Count",
                    color="Status",
                    color_discrete_map={
                        "Completed": "green",
                        "Failed": "red",
                        "Null": "gray"
                    },
                    hole=0.3
                )
                fig2.update_traces(textinfo="none")  # Hide labels
                st.plotly_chart(fig2, use_container_width=True)

            with table_col:
                st.markdown("### Status Counts")
                st.dataframe(summary_df, use_container_width=True)
                st.markdown("### Update Date Counts")
                st.dataframe(updated_count_df, use_container_width=True)

        else:
            st.info("No data available for the selected source.")
    else:
        st.warning(f"Required columns '{status_column}' or '{updated_date_column}' not found in data.")

    st.markdown("### Table Filters")
    col2, col3 = st.columns([1, 1])
    with col2:
        status_filter = st.selectbox("Filter by Status (for Table only)", ["All", "Completed", "Failed", "Null"])

    with col3:
        outcodes = sorted(df["outcode"].dropna().unique().tolist()) if "outcode" in df.columns else []
        selected_outcode = st.selectbox("Filter by OUTCODE (for Table only)", ["All"] + outcodes)

    st.markdown("Filtered Data Table")
    filtered_df = df.copy()

    if status_column in filtered_df.columns:
        filtered_df[status_column] = filtered_df[status_column].fillna("").astype(str).str.strip().str.lower()

        if status_filter != "All":
            if status_filter.lower() == "null":
                filtered_df = filtered_df[filtered_df[status_column] == ""]
            else:
                filtered_df = filtered_df[filtered_df[status_column].str.contains(status_filter.lower(), na=False)]

    if selected_outcode != "All" and "outcode" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["outcode"] == selected_outcode]

    source_prefix = selected_source + "_"
    columns_to_keep = [col for col in filtered_df.columns if col.startswith(source_prefix)]
    if "outcode" in filtered_df.columns:
        columns_to_keep.append("outcode")

    filtered_df = filtered_df[columns_to_keep]

    datetime_cols = [col for col in filtered_df.columns if col.endswith('_updated_datetime')]
    for col in datetime_cols:
        filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')

    st.dataframe(filtered_df.reset_index(drop=True))
