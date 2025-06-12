import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from engine import render_engine_dashboard

st.set_page_config(page_title="Auto Pal", layout="wide")

# ---------- MONGODB CONNECTION ----------
@st.cache_resource
def get_mongo_client():
    return MongoClient(" ") # your MongoDB connection string

client = get_mongo_client()
db = client[" "] # your MongoDB database name
collection = db[" "] # your MongoDB collection name
log_collection = db[" "] # your MongoDB Update collection name

st.sidebar.image("logo.png", width=250, use_container_width=False)
st.sidebar.markdown(
    "<h1 style='font-size:35px; color:white;'>Auto Pal</h1>",
    unsafe_allow_html=True
)
page = st.sidebar.radio("Go to", ["Validation Tool", "Validation Convertible Status", "Validation Dashboard","Engine Dashboard"], index=0)

# --------------------- VALIDATION TOOL ---------------------
if page == "Validation Tool":
    st.markdown("<h1 style='text-align:center;'>Validation Tool</h1>", unsafe_allow_html=True)

    col1, col2,col3 = st.columns([1,1, 1])
    with col1:
        ai = st.selectbox("AI", ["Potential", "Under Value"])
    with col2:
        source = st.selectbox("Source", ["rightmove", "zoopla"])
    with col3:
        property_type = st.selectbox("Property type", ["filter excluded","flat", "house", "hmo"])

    col4,col5, col6 = st.columns([1,1, 1])
    with col4:
        convertible_status = st.selectbox("Convertible status", [1, 2, 3, 4])
    with col5:
        date_from = st.date_input("Date Range From")
    with col6:
        date_to = st.date_input("Date Range To")

    if st.button("Search"):
        date_from_dt = datetime.combine(date_from, datetime.min.time())
        date_to_dt = datetime.combine(date_to, datetime.max.time())
        date_from_str = date_from_dt.strftime("%Y-%m-%d")
        date_to_str = date_to_dt.strftime("%Y-%m-%d")
        # source_str = source.lower()
        query = {
            "convertible_status": convertible_status,
            "first_listed": {
                "$gte": date_from_str,
                "$lte": date_to_str
            },
            "source": source,
            "filter_property_type": property_type, 
            "update_timestamp": {"$exists": False}
        }

        logged_pids = log_collection.distinct("pid")
        query["pid"] = {"$nin": logged_pids}        
        results = list(collection.find(query))

        if not results:
            st.warning("No records found for the selected criteria.")
            st.session_state.results = []
            st.session_state.show_counts = True
        else:
            st.session_state.current_index = 0
            st.session_state.results = results
            st.session_state.show_counts = False

            # Save filters for dashboard
            st.session_state.dashboard_filters = {
                "source": source,
                "convertible_status": convertible_status,
                "filter_property_type": property_type,
                "date_from": date_from,
                "date_to": date_to
            }

    if "results" in st.session_state and st.session_state.results:
        current = st.session_state.current_index
        total = len(st.session_state.results)

        if 0 <= current < total:
            doc = st.session_state.results[current]

            col18, col19 = st.columns([1, 1])
            with col18:
                col_nav_spacer1, col_prev, col_nav_spacer2, col_next, col_nav_spacer3 = st.columns([2, 1, 1, 1, 2])
                with col_prev:
                    if st.button("Previous"):
                        if st.session_state.current_index > 0:
                            st.session_state.current_index -= 1
                            st.rerun()
                with col_next:
                    if st.button("Next"):
                        if st.session_state.current_index < total - 1:
                            st.session_state.current_index += 1
                            st.rerun()

                col_img_spacer1, col_img_spacer2, col_img, col_img_spacer3, col_img_spacer4= st.columns([1,1,1,30,30])
                image_list = doc.get("prop_flp", [])

                if isinstance(image_list, list) and image_list:
                    # Initialize the image index in session state if not present or not for current pid
                    if "img_idx" not in st.session_state or st.session_state.get("img_pid") != doc.get("pid"):
                        st.session_state.img_idx = 0
                        st.session_state.img_pid = doc.get("pid")

                    # Render image buttons outside of nested column
                    st.markdown("### Select Image:")
                    btn_cols = st.columns(len(image_list))
                    for i in range(len(image_list)):
                        with btn_cols[i]:
                            if st.button(f"Image {i+1}"):
                                st.session_state.img_idx = i
                                st.session_state.img_pid = doc.get("pid")
                                st.rerun()

                    # Show selected image inside col_img (still allowed)
                    with col_img:
                        selected_idx = st.session_state.img_idx
                        selected_img_url = image_list[selected_idx]
                        st.image(selected_img_url, caption=f"Image {current + 1} of {total}", width=700)
                else:
                    with col_img:
                        st.markdown(
                            "<div style='color: red; font-weight: bold;'>No images available for this property.</div>",
                            unsafe_allow_html=True
                        )
                   
            with col19:
                st.write(f"PID : {doc['pid']}")
                if ai == "Potential":
                    st.write(f"Convertible Status : {doc.get('convertible_status', 'N/A')}")
                    status_labels = {
                        1: "1-Not Convertible",
                        2: "2-Convertible"
                    }
                    current_status = str(doc.get("convertible_status", 1))
                    status_options = [f"{k}-{v.split('-', 1)[1]}" for k, v in status_labels.items()]
                    formatted_current = status_labels.get(int(current_status), f"{current_status}-Unknown")
                    if formatted_current not in status_options:
                        status_options.insert(0, formatted_current)

                    convertible_status_input = st.selectbox(
                        "Convertible status",
                        status_options,
                        index=status_options.index(formatted_current)
                    )
                    status_val = int(convertible_status_input.split("-")[0])
                    comment = st.text_input("Comment")
                    validate_col, submit_col,spacer = st.columns([1, 1,5])
                    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Common log_data fields
                    log_data = {
                        "pid": doc.get("pid"),
                        "source": source,
                        "prop_flp": doc.get("prop_flp", ""),
                        "ai_type": ai,
                        "update_timestamp": current_date_str,
                        "comment": comment
                    }

                    # Append based on AI type
                    if ai == "Potential":
                        status_val = int(convertible_status_input.split("-")[0])
                        log_data["convertible_status"] = status_val

                    with validate_col:
                        if st.button("Validate"):
                            log_collection.update_one(
                                {"pid": doc["pid"]},
                                {"$set": log_data},
                                upsert=True
                            )
                            df = pd.DataFrame([log_data])
                            df.to_csv("updation_data.csv", mode='a', header=not os.path.isfile("updation_data.csv"), index=False)
                            st.success("Entry logged as validated with no changes.")
                            st.write(log_data)

                    with submit_col:
                        if st.button("Submit"):
                            update_fields = {"update_timestamp": current_date_str}
                            if ai == "Potential":
                                update_fields["convertible_status"] = log_data["convertible_status"]
                            collection.update_one(
                                {"_id": doc["_id"]},
                                {"$set": update_fields}
                            )
                            log_collection.insert_one(log_data)
                            df = pd.DataFrame([log_data])
                            df.to_csv("updation_data.csv", mode='a', header=not os.path.isfile("updation_data.csv"), index=False)
                            st.success("Database updated and logged successfully.")
                            st.write(log_data)                    

                else:
                    st.write(f"Value Prediction : {doc.get('value_prediction', 'N/A')}")
                    value_prediction_options = ["0", "1"]
                    existing_prediction = str(doc.get("value_prediction", "0"))
                    if existing_prediction not in value_prediction_options:
                        value_prediction_options.insert(0, existing_prediction)
                    value_prediction_input = st.selectbox(
                        "Value Prediction",
                        value_prediction_options,
                        index=value_prediction_options.index(existing_prediction)
                    )
                    highlight_options = [
                        "The bathroom is located only on the ground floor, which may impact the property's overall value.",
                        "The property has a short lease, which could affect the property's value."
                    ]       
                    # Get existing highlights from the document
                    existing_highlights = doc.get("undervalue_highlights", [])
                    # Ensure it's a list (in case it's stored as a single string)
                    if not isinstance(existing_highlights, list):
                        existing_highlights = [existing_highlights]
                    # Add any missing existing values to the options
                    for h in existing_highlights:
                        if h not in highlight_options:
                            highlight_options.insert(0, h)
                    # Let the user pick multiple highlights
                    selected_highlights = st.multiselect(
                        "Undervalue Highlights",
                        options=highlight_options,
                        default=existing_highlights
                    )
                    comment = st.text_input("Comment")

                    validate_col, submit_col,spacer = st.columns([1, 1,5])
                    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Common log_data fields
                    log_data = {
                        "pid": doc.get("pid"),
                        "source": source,
                        "prop_flp": doc.get("prop_flp", ""),
                        "ai_type": ai,
                        "update_timestamp": current_date_str,
                        "comment": comment
                    }

                    # Append based on AI type
                    if ai == "Under Value":
                        value_prediction = int(value_prediction_input)
                        log_data["value_prediction"] = value_prediction
                        log_data["undervalue_highlights"] = selected_highlights

                    with validate_col:
                        if st.button("Validate"):
                            log_collection.update_one(
                                {"pid": doc["pid"]},
                                {"$set": log_data},
                                upsert=True
                            )
                            df = pd.DataFrame([log_data])
                            df.to_csv("updation_data.csv", mode='a', header=not os.path.isfile("updation_data.csv"), index=False)
                            st.success("Entry logged as validated with no changes.")
                            st.write(log_data)

                    with submit_col:
                        if st.button("Submit"):
                            update_fields = {"update_timestamp": current_date_str}
                            if ai == "Under Value":
                                update_fields["value_prediction"] = value_prediction
                                update_fields["undervalue_highlights"] = selected_highlights

                            collection.update_one(
                                {"_id": doc["_id"]},
                                {"$set": update_fields}
                            )
                            log_collection.insert_one(log_data)
                            df = pd.DataFrame([log_data])
                            df.to_csv("updation_data.csv", mode='a', header=not os.path.isfile("updation_data.csv"), index=False)
                            st.success("Database updated and logged successfully.")
                            st.write(log_data)

# --------------------- CONVERTIBLE STATUS CHART ---------------------
elif page == "Validation Convertible Status":
    st.markdown("<h4 style='text-align: center;'>Validation Convertible Status</h4>", unsafe_allow_html=True)

    # @st.cache_data(ttl=10) 
    def get_status_counts():
        docs = collection.find({"update_timestamp": {"$exists": False}}, {"convertible_status": 1})
        counts = {}
        for doc in docs:
            status = doc.get("convertible_status")
            if status is not None:
                counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items()))

    counts = get_status_counts()
    labels = [str(k) for k in counts.keys()]
    sizes = list(counts.values())

    colors = plt.cm.Paired.colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        explode=[0.05] * len(sizes),
        wedgeprops=dict(width=0.4)
    )

    legend_labels = [f"Status {label}: {count}" for label, count in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        title="Convertible Status",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        fontsize=6,
        title_fontsize=7,
        frameon=True,
        ncol=1
    )

    ax.axis('equal')
    fig.tight_layout(pad=1)
    st.pyplot(fig)

# --------------------- VALIDATION DASHBOARD ---------------------
elif page == "Validation Dashboard":
    st.subheader("Validation Dashboard Overview")
    if "dashboard_filters" in st.session_state:
        filters = st.session_state.dashboard_filters

        date_from = filters["date_from"].strftime('%Y-%m-%d')
        date_to =filters["date_to"].strftime('%Y-%m-%d')
        source = filters["source"]
        convertible_status = int(filters["convertible_status"])
        property_type = filters["filter_property_type"]

        query = {
            "convertible_status": convertible_status,
            "first_listed": {
                "$gte": date_from,
                "$lte": date_to
            },
            "source": source,
            "filter_property_type": property_type, 
            "update_timestamp": {"$exists": False}
        }

        data = list(collection.find(query))

        if not data:
            st.warning("No data found for the given filters.")
            st.stop()

        df = pd.DataFrame(data)

        df["first_listed"] = pd.to_datetime(df["first_listed"], errors="coerce")
        df["bedroom"] = pd.to_numeric(df.get("bedroom", 0), errors="coerce").fillna(0)
        df["convertible_status"] = df["convertible_status"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Property Type by Count")
            if "filter_property_type" in df.columns:
                fig = px.histogram(
                    df,
                    x="filter_property_type",
                    color="filter_property_type",
                    title="Property Type Count",
                    labels={"filter_property_type": "Property Type"}
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Bedroom Distribution")
            fig = px.box(
                df,
                y="bedroom",
                title="Bedroom Count Distribution",
                labels={"bedroom": "Bedrooms"}
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Listings Over Time")
            if not df.empty:
                time_series = df.groupby(df["first_listed"].dt.date).size().reset_index(name="Count")
                time_series["first_listed"] = pd.to_datetime(time_series["first_listed"])
                fig = px.line(
                    time_series,
                    x="first_listed",
                    y="Count",
                    title="Listings by Load Date",
                    labels={"first_listed": "Date", "Count": "Listings"}
                )
                st.plotly_chart(fig, use_container_width=True)
        with col4:
            if "postcode" in df.columns and "bedroom" in df.columns:
                st.subheader("Postcode vs Bedroom Heatmap")
                heatmap_data = pd.crosstab(df["postcode"], df["bedroom"].astype(int))
                st.dataframe(heatmap_data.style.background_gradient(cmap="Blues"))
    else:
        st.warning("No filters received from Validation Tool page. Please navigate through the Validation Tool page.")

# --------------------- ENGINE DASHBOARD ---------------------
elif page == "Engine Dashboard":
    render_engine_dashboard()