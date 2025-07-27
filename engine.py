import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px

def render_engine_dashboard():
    # st.set_page_config(page_title="Engine Dashboard", layout="wide")

    # ---------- MongoDB Connection ----------
    @st.cache_resource
    def connect_to_mongo():
        mongo_url = " "  
        client = MongoClient(mongo_url)
        db = client[" "]
        collection = db[" "]
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

    st.markdown(f"<h1 style='text-align:center;'>Engine Dashboard</h1>", unsafe_allow_html=True)

    # ---------- Valid Source Status Fields ----------
    status_sources = [
        "onthemarket_ai",
        # "onthemarket_commercial",
        "onthemarket_let", "onthemarket_sale",
        "rightmove_ai","rightmove_commercial", "rightmove_let", "rightmove_sale",
        "zoopla_ai","zoopla_commercial", "zoopla_let", "zoopla_sale"
    ]

    # ---------- UI Filter: Select Source for Chart ----------
    col1 = st.columns(1)[0]
    with col1:
        selected_source = st.selectbox("Select Source (for Chart & Table)", status_sources)

    # ---------- Pie Chart and Summary Table ----------
    st.subheader(f"Status Overview for {selected_source}")
    status_column = f"{selected_source}_status"
    updated_date_column = f"{selected_source}_updated_date"

    if status_column in df.columns and updated_date_column in df.columns:
        pie_df = df.copy()

        # Normalize and clean up the status column
        pie_df[status_column] = pie_df[status_column].fillna("").astype(str).str.strip().str.lower()
        pie_df[status_column] = pie_df[status_column].replace("", "null")

        # Count all unique status values
        status_counts = pie_df[status_column].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        total = status_counts["Count"].sum()

        if total > 0:
            status_counts["Percentage"] = (status_counts["Count"] / total * 100).round(2).astype(str) + "%"
            summary_df = status_counts

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
                    color_discrete_map=None,
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
    col2, col3, col4 = st.columns([1, 1, 1])
    with col2:
        status_values = df[status_column].fillna("").astype(str).str.strip().str.lower().replace("", "null").unique().tolist()
        status_values = sorted(status_values)
        status_filter = st.selectbox("Filter by Status (for Table only)", ["All"] + status_values)
    with col3:
        outcodes = sorted(df["outcode"].dropna().unique().tolist()) if "outcode" in df.columns else []
        selected_outcode = st.selectbox("Filter by OUTCODE (for Table only)", ["All"] + outcodes)
    with col4:
        date_col = f"{selected_source}_updated_date"
        if date_col in df.columns:
            # Convert mixed formats to datetime safely
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # Extract only the date part
            df["filter_date"] = df[date_col].dt.date
            # Generate unique dates for selection
            unique_dates = sorted(df["filter_date"].dropna().unique())
            if unique_dates:
                selected_date = st.selectbox("Filter by Updated Date", ["All"] + [str(d) for d in unique_dates])
            else:
                selected_date = "All"
        else:
            selected_date = "All"

    st.markdown("Filtered Data Table")
    filtered_df = df.copy()

    if status_column in filtered_df.columns:
        filtered_df[status_column] = filtered_df[status_column].fillna("").astype(str).str.strip().str.lower()

        if status_filter != "All":
            filtered_df[status_column] = filtered_df[status_column].fillna("").astype(str).str.strip().str.lower()
            filtered_df[status_column] = filtered_df[status_column].replace("", "null")
            filtered_df = filtered_df[filtered_df[status_column] == status_filter]

    if selected_outcode != "All" and "outcode" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["outcode"] == selected_outcode]

    source_prefix = selected_source + "_"
    columns_to_keep = [col for col in filtered_df.columns if col.startswith(source_prefix)]
    if "outcode" in filtered_df.columns:
        columns_to_keep.append("outcode")

    filtered_df = filtered_df[columns_to_keep]
    if date_col in filtered_df.columns:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")
        filtered_df["filter_date"] = filtered_df[date_col].dt.date

    # Final row filter by selected date
    if selected_date != "All" and "filter_date" in filtered_df.columns:
        selected_date_obj = pd.to_datetime(selected_date).date()
        filtered_df = filtered_df[filtered_df["filter_date"] == selected_date_obj]

    # Now filter columns — include only the relevant ones and EXCLUDE _updated_datetime
    source_prefix = selected_source + "_"
    updated_date_column = f"{selected_source}_updated_datetime"
    date_col = f"{selected_source}_updated_date"  # Redefine here for safety

    # columns_to_keep = [
    #     col for col in filtered_df.columns 
    #     if col.startswith(source_prefix) and col != updated_date_column
    # ]

    columns_to_keep = [col for col in filtered_df.columns if col.startswith(source_prefix)]

    for extra in ["outcode", date_col]:
        if extra in filtered_df.columns and extra not in columns_to_keep:
            columns_to_keep.append(extra)

    try:
        columns_to_keep.remove(f'{source_prefix}updated_datetime')
    except:
        pass
    filtered_df = filtered_df[columns_to_keep]

    # Convert selected_source_updated_date to just date (no time)
    if date_col in filtered_df.columns:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce").dt.date

    datetime_cols = [col for col in filtered_df.columns if col.endswith('_updated_datetime')]
    for col in datetime_cols:
        filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')

    st.dataframe(filtered_df.reset_index(drop=True))
