import os
import boto3
import base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import streamlit as st
import plotly.express as px
from datetime import datetime
from pymongo import MongoClient
import matplotlib.pyplot as plt
from datetime import date, timedelta
from engine import render_engine_dashboard
# from monthly_task import render_monthly_task

#-----------S3 Bucket----------
def load_image_from_s3(key):
  
    bucket = ''
    access_key = ''
    secret_access_key = ''
    # S3 = boto3.resource( 's3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key )
    s3 = boto3.client( 's3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key )
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response['Body'].read()))
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return None

# ---------- LOGIN HANDLER ----------
USER_CREDENTIALS = {
    "admin": {"password": "Marker@281"},
    "sidharth": {"password": "Sidharth@123"},
    "sachin": {"password": "Sachin@123"},    
    "jisha": {"password": "Jisha@123"},
    "vipin": {"password": "Vipin@123"}
}
 
# ---------- LOGIN HANDLER ----------
def login():
    st.set_page_config(page_title="Login", layout="centered")
    # Center the logo and title using Streamlit components
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{}' width='350'/>
        </div>
        """.format(base64.b64encode(open("logo.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )
    st.markdown(
        "<h1 style='text-align: center; margin-top: 0;'><b>Login to Auto Pal</b></h1>",
        unsafe_allow_html=True
    )
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = USER_CREDENTIALS.get(username)
        if user and password == user["password"]:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
 
# ---------- SESSION INIT ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
 
# ---------- REDIRECT TO LOGIN IF NOT AUTHENTICATED ----------
if not st.session_state["logged_in"]:
    login()
    st.stop()

st.set_page_config(page_title="Auto Pal", layout="wide")

# ---------- MONGODB CONNECTION ----------
@st.cache_resource
def get_mongo_client():
    return MongoClient("")

def get_collection_by_session(ai_type, source, client):
    ai = ai_type.lower()
    src = source.lower()
    if ai == "potential":
        if src == "rightmove":
            db = client["rightmove_data"]
            return db["rightmove_outcodes_ai"], db["rightmove_ai_update_logs"]
        elif src == "zoopla":
            db = client["zoopla_data"]
            return db["zoopla_outcodes_ai"], db["zoopla_ai_update_logs"]
    elif ai == "under value":
        if src == "rightmove":
            db = client["rightmove_data"]
            return db["rightmove_outcodes_uv"], db["rightmove_uv_update_logs"]
        elif src == "zoopla":
            db = client["zoopla_data"]
            return db["zoopla_outcodes_uv"], db["zoopla_uv_update_logs"]
    return None, None

# logo
st.sidebar.image("logo.png", width=250, use_container_width=False)

# Show current login status
if "username" in st.session_state:
    st.sidebar.markdown(
        f"Logged in as: <b>{st.session_state['username']}</b>",
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    "<h1 style='font-size:35px; color:skyblue;'>Auto Pal</h1>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Go to",
    [
        "**Validation Tool**",
        "**Validation Review**",
        "**Validation Status**",
        "**Validation Dashboard**",
        "**Engine Dashboard**"
    ],
    index=0
)
# Remove markdown bold for logic
page = page.replace("**", "")

# Manual refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()  # Clear cached data
    st.session_state["force_refresh"] = True
    st.rerun()  

client = get_mongo_client()
# --------------------- VALIDATION TOOL ---------------------
if page == "Validation Tool":
    st.markdown("<h1 style='text-align:center;'>Validation Tool</h1>", unsafe_allow_html=True)

     # Set defaults only once
    if "ai" not in st.session_state:
        st.session_state.ai = "Potential"
    if "source" not in st.session_state:
        st.session_state.source = "rightmove"
    if "property_type" not in st.session_state:
        st.session_state.property_type = "filter excluded"
    if "convertible_status" not in st.session_state:
        st.session_state.convertible_status = 1
    if "value_prediction" not in st.session_state:
        st.session_state.value_prediction = "0"
    if "date_from" not in st.session_state:
        st.session_state.date_from = datetime.today().date()
    if "date_to" not in st.session_state:
        st.session_state.date_to = datetime.today().date()
 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.session_state.ai = st.selectbox("AI", ["Potential", "Under Value"], index=["Potential", "Under Value"].index(st.session_state.ai))
    with col2:
        st.session_state.source = st.selectbox("Source", ["rightmove", "zoopla"], index=["rightmove", "zoopla"].index(st.session_state.source))
    with col3:
        st.session_state.property_type = st.selectbox("Property type", ["filter excluded", "flat", "house", "hmo"], index=["filter excluded", "flat", "house", "hmo"].index(st.session_state.property_type))
 
    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        if st.session_state.ai == "Potential":
            st.session_state.convertible_status = st.selectbox("Convertible status", [1, 2, 3, 4], index=[1, 2, 3, 4].index(st.session_state.get("convertible_status", 1)))
        else:
            st.session_state.value_prediction = st.selectbox("Value Prediction", ["0", "1"], index=["0", "1"].index(st.session_state.get("value_prediction", "0")))
    with col5:
        st.session_state.date_from = st.date_input("Date Range From", st.session_state.date_from)
    with col6:
        st.session_state.date_to = st.date_input("Date Range To", st.session_state.date_to)

    ai_type = st.session_state.ai.lower()
    source = st.session_state.source.lower()

    collection, log_collection = get_collection_by_session(ai_type=st.session_state.ai, source=st.session_state.source, client=client)
    # print(collection ,log_collection)
    if collection is None:
        st.error("Invalid AI type or source. Cannot connect to the correct collection.")
        st.stop()

    if st.button("Search"):
        start_datetime = datetime.combine(st.session_state.date_from, datetime.min.time())
        end_datetime = datetime.combine(st.session_state.date_to, datetime.max.time())

        query = {
            "etl_load_timestamp": {"$gte": start_datetime, "$lte": end_datetime},
            "update_timestamp": {"$exists": False}
        }
        if st.session_state.ai == "Potential":
            query["convertible_status"] = st.session_state.convertible_status
            query["filter_property_type"] = st.session_state.property_type
        else:
            query["value_prediction"] = int(st.session_state.value_prediction)
            selected_type = st.session_state.property_type.lower()
            if selected_type == "house":
                house_variants = [
                    "semi detached villa", "semi detached", "semi detached house",
                    "terrace", "terraced", "end terrace", "end terraced", "end of terrace",
                    "end of terraced", "end of terrace house", "end of terraced house", "end terrace house",
                    "end terraced house", "terraced house", "terrace house", "terraced villa",
                    "detached house", "detached villa", "link detached house", "link detached", "detached",
                    "house", "houses", "manor house", "cluster house", "mews", "cottage",
                    "smallholding", "equestrian", "leisure", "villa"
                ]
                query["type"] = {"$in": house_variants}
            else:
                query["type"] = selected_type

        logged_pids = log_collection.find({"ai_type": st.session_state.ai}, {"pid": 1})
        validated_pids = {doc["pid"] for doc in logged_pids}
        query["pid"] = {"$nin": list(validated_pids)}
        print(query)
        results = list(collection.find(query))
        # print(results)

        if not results:
            st.warning("No records found for the selected criteria.")
            st.session_state.results = []
            st.session_state.show_counts = True
        else:
            st.session_state.current_index = 0
            st.session_state.results = results
            st.session_state.show_counts = False

    if "results" in st.session_state and st.session_state.results:
        current = st.session_state.current_index
        total = len(st.session_state.results)

        if 0 <= current < total:
            doc = st.session_state.results[current]
            # Reset form inputs if we're on a new PID
            if st.session_state.get("last_pid") != doc["pid"]:
                st.session_state.last_pid = doc["pid"]

                # Reset for Potential AI
                if st.session_state.ai == "Potential":
                    current_status = str(doc.get("convertible_status", 1))
                    status_labels = {1: "1-Not Convertible", 2: "2-Convertible"}
                    formatted_current = status_labels.get(int(current_status), f"{current_status}-Unknown")
                    st.session_state.convertible_status_input = formatted_current
                    st.session_state.comment_input = doc.get("comment", "")

                # Reset for Under Value AI
                elif st.session_state.ai == "Under Value":
                    st.session_state.value_prediction_input = str(doc.get("value_prediction", "0"))
                    highlights = doc.get("undervalue_highlights", [])
                    if not isinstance(highlights, list):
                        highlights = [highlights]
                    st.session_state.selected_highlights = highlights
                    st.session_state.comment_input = doc.get("comment", "")

            col18, col19 = st.columns([1, 1])
            with col18:
                if st.session_state.ai == "Potential":
                    image_container = st.container()
                    with image_container:
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        pid = doc.get("pid")
                        property_type = doc.get("filter_property_type")

                        if property_type == 'flat':
                            s3_key = f"{source}/images/{pid}/image-kitchen-hightlight/{pid}.jpg"
                        elif property_type == 'house':
                            s3_key = f"{source}/images/{pid}/image-bathroom-highlight/{pid}.jpg"
                        elif property_type == 'hmo':
                            s3_key = f"{source}/images/{pid}/image-hmo-highlight/{pid}.jpg"
                        else:
                            s3_key = None

                        if s3_key:
                            image = load_image_from_s3(s3_key)
                            st.markdown(
                                f"<h5 style='text-align: center;'>Image {current + 1} of {total}</h5>",
                                unsafe_allow_html=True
                            )
                            if image:
                                st.image(image, use_container_width=True)
                            else:
                                st.warning("S3 image could not be loaded.")
                        else:
                            st.warning("No valid image key generated for this property.")
                        st.markdown("</div>", unsafe_allow_html=True)

                elif st.session_state.ai == "Under Value":
                    if source == "rightmove":
                        db = client["rightmove_data"]
                        sale_collection = db["rightmove_sales"]
                        property_data = sale_collection.find_one({"pid": doc.get("pid")})

                        if property_data:
                            image_list = property_data.get("prop_flp", [])
                        else:
                            image_list = []

                        if isinstance(image_list, list) and image_list:
                            if "img_idx" not in st.session_state or st.session_state.get("img_pid") != doc.get("pid"):
                                st.session_state.img_idx = 0
                                st.session_state.img_pid = doc.get("pid")
                            st.markdown("### Select Image:")
                            btn_cols = st.columns(len(image_list))
                            for i in range(len(image_list)):
                                with btn_cols[i]:
                                    if st.button(f"Image {i+1}"):
                                        st.session_state.img_idx = i
                                        st.session_state.img_pid = doc.get("pid")
                                        st.rerun()
                            selected_img_url = image_list[st.session_state.img_idx]
                            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                            st.markdown(
                                f"<h5 style='text-align: center;'>Image {current + 1} of {total}</h5>",
                                unsafe_allow_html=True
                            )
                            st.image(selected_img_url, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No images found in the Under Value rightmove source.")

                    elif source == "zoopla":
                        db = client["zoopla_data"]
                        sale_collection = db["zoopla_sales"]
                        property_data = sale_collection.find_one({"pid": doc.get("pid")})

                        if property_data:
                            image_list = property_data.get("prop_flp", [])
                        else:
                            image_list = []

                        if isinstance(image_list, list) and image_list:
                            if "img_idx" not in st.session_state or st.session_state.get("img_pid") != doc.get("pid"):
                                st.session_state.img_idx = 0
                                st.session_state.img_pid = doc.get("pid")
                            st.markdown("### Select Image:")
                            btn_cols = st.columns(len(image_list))
                            for i in range(len(image_list)):
                                with btn_cols[i]:
                                    if st.button(f"Image {i+1}"):
                                        st.session_state.img_idx = i
                                        st.session_state.img_pid = doc.get("pid")
                                        st.rerun()
                            selected_img_url = image_list[st.session_state.img_idx]
                            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                            st.markdown(
                                f"<h5 style='text-align: center;'>Image {current + 1} of {total}</h5>",
                                unsafe_allow_html=True
                            )
                            st.image(selected_img_url, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No images found in the Under Value zoopla source.")

            with col19:
                collection, log_collection = get_collection_by_session(ai_type=st.session_state.ai, source=st.session_state.source, client=client)

                st.write(f"PID : {doc['pid']}")
                doc_key_prefix = f"doc_{st.session_state.current_index}_{doc.get('pid')}"
                if st.session_state.ai == "Potential":
                    st.write(f"Convertible Status : {doc.get('convertible_status', 'N/A')}")
                    status_labels = {1: "1-Not Convertible", 2: "2-Convertible"}
                    current_status = str(doc.get("convertible_status", 1))
                    status_options = [f"{k}-{v.split('-', 1)[1]}" for k, v in status_labels.items()]
                    formatted_current = status_labels.get(int(current_status), f"{current_status}-Unknown")
                    if formatted_current not in status_options:
                        status_options.insert(0, formatted_current)
                    convertible_status_input = st.selectbox("Convertible status",status_options,index=status_options.index(formatted_current),key=f"{doc_key_prefix}_convertible_status")
                    status_val = int(convertible_status_input.split("-")[0])
                    comment = st.text_input("Comment", key=f"{doc_key_prefix}_comment")
                else:
                    status_val = int(doc.get("convertible_status", 1))
                    st.write(f"Value Prediction : {doc.get('value_prediction', 'N/A')}")
                    value_prediction_options = ["0", "1"]
                    existing_prediction = str(doc.get("value_prediction", "0"))
                    if existing_prediction not in value_prediction_options:
                        value_prediction_options.insert(0, existing_prediction)
                    value_prediction_input = st.selectbox("Value Prediction",value_prediction_options,index=value_prediction_options.index(existing_prediction),key=f"{doc_key_prefix}_value_prediction")
                    highlight_options = [
                        "The bathroom is located only on the ground floor, which may impact the property's overall value.",
                        "The property has a short lease, which could affect the property's value."
                    ]
                    existing_highlights = doc.get("undervalue_highlights", [])
                    if not isinstance(existing_highlights, list):
                        existing_highlights = [existing_highlights]
                    for h in existing_highlights:
                        if h not in highlight_options:
                            highlight_options.insert(0, h)
                    selected_highlights = st.multiselect("Undervalue Highlights",options=highlight_options,default=existing_highlights,key=f"{doc_key_prefix}_highlights")
                    comment = st.text_input("Comment", key=f"{doc_key_prefix}_comment")

                submit_col, col_nav_spacer2, col_prev, col_nav_spacer3, col_next = st.columns([1, 3,1, 1, 1])
                current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_data = {
                    "pid": doc.get("pid"),
                    "source": st.session_state.source,
                    "prop_flp": doc.get("prop_flp", ""),
                    "ai_type": st.session_state.ai,
                    "update_timestamp": current_date_str,
                    "comment": comment,
                    "username": st.session_state.get("username", "unknown")
                }

                if st.session_state.ai == "Potential":
                    log_data["filter_property_type"] = st.session_state.property_type
                    log_data["convertible_status"] = status_val
                elif st.session_state.ai == "Under Value":
                    log_data["type"] = doc.get("type", "Unknown")
                    log_data["value_prediction"] = int(value_prediction_input)
                    log_data["undervalue_highlights"] = selected_highlights


                main_update_fields = {}
                if st.session_state.ai == "Potential":
                    main_update_fields["convertible_status"] = status_val
                elif st.session_state.ai == "Under Value":
                    main_update_fields["value_prediction"] = int(value_prediction_input)
                    main_update_fields["undervalue_highlights"] = selected_highlights
                with submit_col:
                    if st.button("Submit"):
                        collection, log_collection = get_collection_by_session(ai_type=st.session_state.ai, source=st.session_state.source, client=client)
                        log_collection.update_one({"pid": doc["pid"], "ai_type": st.session_state.ai}, {"$set": log_data}, upsert=True)
                        update_fields = log_data.copy()
                        collection.update_one({"_id": doc["_id"]}, {"$set": main_update_fields})
                        pd.DataFrame([log_data]).to_csv("validation_data.csv", mode='a', header=not os.path.isfile("validation_data.csv"), index=False)
                        st.success("Database updated and logged successfully.")
                        if st.session_state.current_index < len(st.session_state.results) - 1:
                            st.session_state.current_index += 1
                            st.rerun()

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

# --------------------- VALIDATION REVIEW ---------------------
elif page == "Validation Review":

    st.markdown("<h1 style='text-align:center;'>Validation Review</h1>", unsafe_allow_html=True)

    if "review_search_triggered" not in st.session_state:
        st.session_state.review_search_triggered = False

    # Load logs from all update logs
    rm_ai_logs = list(client["rightmove_data"]["rightmove_ai_update_logs"].find({"update_timestamp": {"$exists": True}}).limit(1000))
    rm_uv_logs = list(client["rightmove_data"]["rightmove_uv_update_logs"].find({"update_timestamp": {"$exists": True}}).limit(1000))
    zp_ai_logs = list(client["zoopla_data"]["zoopla_ai_update_logs"].find({"update_timestamp": {"$exists": True}}).limit(1000))
    zp_uv_logs = list(client["zoopla_data"]["zoopla_uv_update_logs"].find({"update_timestamp": {"$exists": True}}).limit(1000))
    df_logs_sample = pd.DataFrame(rm_ai_logs + rm_uv_logs + zp_ai_logs + zp_uv_logs)

    # Filter UI
    col1, col2, col3 = st.columns(3)
    with col1:
        usernames = sorted(df_logs_sample["username"].dropna().unique()) if not df_logs_sample.empty else []
        selected_user = st.selectbox("Select user", ["All"] + usernames)
    with col2:
        ai_types = sorted(df_logs_sample["ai_type"].dropna().unique()) if not df_logs_sample.empty else []
        selected_ai = st.selectbox("Select AI Type", ["All"] + ai_types, key="review_ai_filter")
    with col3:
        sources = sorted(df_logs_sample["source"].dropna().unique()) if not df_logs_sample.empty else ["rightmove", "zoopla"]
        selected_source = st.selectbox("Select Source", ["All"] + sources, key="review_source_filter")

    col4, col5, col6 = st.columns(3)
    with col4:
        property_types = sorted(df_logs_sample["filter_property_type"].dropna().unique()) if not df_logs_sample.empty else []
        selected_property_type = st.selectbox("Property Type", ["All"] + property_types, key="review_property_type_filter")
    with col5:
        if selected_ai == "Potential":
            convertible_statuses = sorted(df_logs_sample["convertible_status"].dropna().astype(int).unique()) if not df_logs_sample.empty else []
            selected_status = st.selectbox("Convertible Status", ["All"] + list(map(str, convertible_statuses)), key="review_cs_filter")
        elif selected_ai == "Under Value":
            value_predictions = sorted(df_logs_sample["value_prediction"].dropna().astype(int).unique()) if not df_logs_sample.empty else []
            selected_prediction = st.selectbox("Value Prediction", ["All"] + list(map(str, value_predictions)), key="review_vp_filter")
    with col6:
        try:
            df_logs_sample["update_timestamp"] = pd.to_datetime(df_logs_sample["update_timestamp"], errors="coerce")
            min_date = df_logs_sample["update_timestamp"].min().date()
            max_date = df_logs_sample["update_timestamp"].max().date()
            if pd.isna(min_date) or pd.isna(max_date):
                min_date = date(2020, 1, 1)
                max_date = date(2030, 1, 1)
        except:
            min_date = date(2020, 1, 1)
            max_date = date(2030, 1, 1)

        selected_date_range = st.date_input(
            "Select Date Range", [min_date, max_date],
            min_value=date(2020, 1, 1),
            max_value=date(2030, 1, 1),
            key="review_date_filter"
        )

    if st.button("Search"):
        if selected_ai == "All" or selected_source == "All":
            st.warning("Please select both Source and AI Type to proceed.")
        else:
            st.session_state.review_search_triggered = True

    if st.session_state.review_search_triggered:
        source = selected_source.lower()
        ai = selected_ai.lower().replace(" ", "")
        if source == "rightmove" and ai == "potential":
            log_collection = client["rightmove_data"]["rightmove_ai_update_logs"]
        elif source == "rightmove" and ai == "undervalue":
            log_collection = client["rightmove_data"]["rightmove_uv_update_logs"]
        elif source == "zoopla" and ai == "potential":
            log_collection = client["zoopla_data"]["zoopla_ai_update_logs"]
        elif source == "zoopla" and ai == "undervalue":
            log_collection = client["zoopla_data"]["zoopla_uv_update_logs"]
        else:
            st.warning("Invalid Source or AI Type combination.")
            st.stop()

        all_logs = list(log_collection.find({"update_timestamp": {"$exists": True}}).sort("update_timestamp", -1))
        if not all_logs:
            st.warning("No logs found.")
            st.stop()

        df_logs = pd.DataFrame(all_logs)
        df_logs["update_timestamp"] = pd.to_datetime(df_logs["update_timestamp"], errors="coerce")
        df_logs["date_only"] = df_logs["update_timestamp"].dt.date

        filtered_logs = df_logs.copy()
        if selected_user != "All":
            filtered_logs = filtered_logs[filtered_logs["username"] == selected_user]
        if selected_property_type != "All":
            filtered_logs = filtered_logs[filtered_logs["filter_property_type"] == selected_property_type]
        if selected_ai == "Potential" and 'selected_status' in locals() and selected_status != "All":
            filtered_logs = filtered_logs[filtered_logs["convertible_status"] == int(selected_status)]
        if selected_ai == "Under Value" and 'selected_prediction' in locals() and selected_prediction != "All":
            filtered_logs = filtered_logs[filtered_logs["value_prediction"] == int(selected_prediction)]
        if selected_date_range:
            start_date, end_date = selected_date_range
            filtered_logs = filtered_logs[
                (filtered_logs["update_timestamp"].dt.date >= start_date) &
                (filtered_logs["update_timestamp"].dt.date <= end_date)
            ]

        if filtered_logs.empty:
            st.warning("No logs match the filter criteria.")
            st.stop()

        logs = filtered_logs.to_dict(orient="records")
        if "review_index" not in st.session_state:
            st.session_state.review_index = 0
        total_logs = len(logs)
        st.session_state.review_index = min(st.session_state.review_index, total_logs - 1)

        current_log = logs[st.session_state.review_index]
        pid = current_log["pid"]
        ai_type = current_log.get("ai_type", "Potential")
        normalized_ai = ai_type.replace(" ", "").lower()
        source = current_log.get("source", "rightmove").lower()

        db = client[f"{source}_data"]
        main_collection = db[f"{source}_outcode_ai"] if normalized_ai == "potential" else db[f"{source}_outcode_uv"]
        main_doc = main_collection.find_one({"pid": pid})
        image_list = main_doc.get("prop_flp", []) if main_doc else []

        if "review_img_idx" not in st.session_state or st.session_state.get("review_pid") != pid:
            st.session_state.review_img_idx = 0
            st.session_state.review_pid = pid

        col_img, col_edit = st.columns([2, 2])
        with col_img:
            if isinstance(image_list, list) and image_list:
                st.markdown("### Select Image:")
                img_buttons = st.columns(len(image_list))
                for i in range(len(image_list)):
                    with img_buttons[i]:
                        if st.button(f"Image {i+1}", key=f"review_img_btn_{i}"):
                            st.session_state.review_img_idx = i
                            st.session_state.review_pid = pid
                            st.rerun()
                selected_img_url = image_list[st.session_state.review_img_idx]
                st.image(selected_img_url, caption=f"Image {st.session_state.review_img_idx + 1} of {len(image_list)}", use_container_width=True)
            else:
                st.warning("No images available.")

        with col_edit:
            st.write(f"**PID**: {pid} ({st.session_state.review_index + 1} of {total_logs})")
            st.write(f"**AI Type**: {ai_type}")
            comment = current_log.get("comment", "")
            new_comment = st.text_input("Comment", value=comment)
            update_data = {
                "comment": new_comment,
                "update_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": current_log.get("username", "unknown"),
                "filter_property_type": current_log.get("filter_property_type", "Unknown"),
                "ai_type": ai_type,
                "source": source
            }

            if normalized_ai == "potential":
                current_status = current_log.get("convertible_status", 1)
                st.markdown(f"Previous Convertible Status: {current_status}")
                new_status = st.selectbox("Convertible Status", [1, 2], index=[1, 2].index(current_status), key=f"status_{pid}_{st.session_state.review_index}")
                update_data["convertible_status"] = new_status
            elif normalized_ai == "undervalue":
                existing_prediction = int(current_log.get("value_prediction", 0))
                st.markdown(f"Previous Value Prediction: {existing_prediction}")
                value_prediction = st.selectbox("Value Prediction", [0, 1], index=[0, 1].index(existing_prediction), key=f"vp_{pid}_{st.session_state.review_index}")
                highlight_options = [
                    "The bathroom is located only on the ground floor, which may impact the property's overall value.",
                    "The property has a short lease, which could affect the property's value."
                ]
                existing_highlights = current_log.get("undervalue_highlights", [])
                if not isinstance(existing_highlights, list):
                    existing_highlights = [existing_highlights]
                st.markdown(f"Previous Highlights: {', '.join(existing_highlights)}")
                selected_highlights = st.multiselect("Highlights", options=highlight_options, default=existing_highlights)
                update_data["value_prediction"] = value_prediction
                update_data["undervalue_highlights"] = selected_highlights

            # --- Control Buttons: Resubmit + Navigation ---
            nav_col1, nav_col2, nav_col3 = st.columns([1, 0.5, 1])
            with nav_col1:
                if st.button("Resubmit", key=f"resubmit_{pid}_{st.session_state.review_index}"):
                    # Get log and main collection
                    if source == "rightmove" and normalized_ai == "potential":
                        log_collection = client["rightmove_data"]["rightmove_ai_update_logs"]
                        main_collection = client["rightmove_data"]["rightmove_outcode_ai"]
                    elif source == "rightmove" and normalized_ai == "undervalue":
                        log_collection = client["rightmove_data"]["rightmove_uv_update_logs"]
                        main_collection = client["rightmove_data"]["rightmove_outcode_uv"]
                    elif source == "zoopla" and normalized_ai == "potential":
                        log_collection = client["zoopla_data"]["zoopla_ai_update_logs"]
                        main_collection = client["zoopla_data"]["zoopla_outcode_ai"]
                    elif source == "zoopla" and normalized_ai == "undervalue":
                        log_collection = client["zoopla_data"]["zoopla_uv_update_logs"]
                        main_collection = client["zoopla_data"]["zoopla_outcode_uv"]
                    else:
                        st.error("Unknown Source + AI combination.")
                        st.stop()

                    log_collection.update_one({"pid": pid}, {"$set": update_data}, upsert=True)
                    main_update_data = {
                        "update_timestamp": update_data["update_timestamp"],
                        "comment": update_data["comment"]
                    }
                    if normalized_ai == "potential":
                        main_update_data["convertible_status"] = update_data["convertible_status"]
                    elif normalized_ai == "undervalue":
                        main_update_data["value_prediction"] = update_data["value_prediction"]
                        main_update_data["undervalue_highlights"] = update_data["undervalue_highlights"]
                    main_collection.update_one({"pid": pid}, {"$set": main_update_data})

                    df = pd.DataFrame([update_data])
                    df.to_csv("validation_data.csv", mode='a', header=not os.path.isfile("validation_data.csv"), index=False)

                    # Go to next
                    if st.session_state.review_index < total_logs - 1:
                        st.success("Updated and moving to next...")
                        st.session_state.review_index += 1
                        st.experimental_rerun()
                    else:
                        st.success("Updated. End of results.")

            with nav_col2:
                if st.button("Previous", key=f"prev_{pid}") and st.session_state.review_index > 0:
                    st.session_state.review_index -= 1
                    st.rerun()

            with nav_col3:
                if st.button("Next", key=f"next_{pid}") and st.session_state.review_index < total_logs - 1:
                    st.session_state.review_index += 1
                    st.rerun()

# --------------------- VALIDATION STATUS ---------------------
elif page == "Validation Status":

    current_user = st.session_state.get("username", "Unknown")
    st.markdown(f"<h1 style='text-align:center;'>{current_user} Validation Statistics</h1>", unsafe_allow_html=True)

    # --- UI Filters ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date_range = st.date_input(
            "Select Date Range",  
            value=[datetime.today() - timedelta(days=30), datetime.today()], 
            help="Select the date range for the properties from main collection"  
        )
    with col2:
        selected_source = st.selectbox("Select Source", options=["rightmove", "zoopla"])
    with col3:
        selected_property_type = st.selectbox("Property Type", options=["All", "filter excluded", "flat", "house", "hmo"])
    with col4:
        selected_ai_type = st.selectbox("AI Type", options=["Potential", "Under Value"])

    if st.button("Search"):
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
            query = {
                "etl_load_timestamp": {
                    "$gte": datetime.combine(date_range[0], datetime.min.time()),
                    "$lte": datetime.combine(date_range[1], datetime.max.time())
                },
                "source": selected_source
            }

            if selected_property_type != "All":
                query["filter_property_type"] = selected_property_type
        else:
            st.warning("Please select both start and end date.")
            st.stop()

        # --- Choose correct main and log collection based on AI type and source ---
        if selected_source == "zoopla":
            if selected_ai_type == "Potential":
                collection_to_query = client["zoopla_data"]["zoopla_outcode_ai"]
                log_collection = client["zoopla_data"]["zoopla_ai_update_logs"]
            else:
                collection_to_query = client["zoopla_data"]["zoopla_outcode_uv"]
                log_collection = client["zoopla_data"]["zoopla_uv_update_logs"]
        else:  # rightmove
            if selected_ai_type == "Potential":
                collection_to_query = client["rightmove_data"]["rightmove_outcode_ai"]
                log_collection = client["rightmove_data"]["rightmove_ai_update_logs"]
            else:
                collection_to_query = client["rightmove_data"]["rightmove_outcode_uv"]
                log_collection = client["rightmove_data"]["rightmove_uv_update_logs"]

        # --- Query main collection to get matching properties ---
        matching_properties = list(collection_to_query.find(query))
        matching_pids = {str(p["pid"]) for p in matching_properties if "pid" in p}

        # --- Get validated PIDs from update_logs ---
        logged_pids = set(log_collection.distinct("pid"))
        logged_pids = {str(pid) for pid in logged_pids}

        # --- Find unvalidated PIDs ---
        unvalidated_pids = matching_pids - logged_pids
        unvalidated_docs = [doc for doc in matching_properties if str(doc.get("pid")) in unvalidated_pids]

        # --- Current User Validated Count ---
        log_query = {
            "update_timestamp": {"$exists": True},
            "source": selected_source,
            "ai_type": selected_ai_type
        }

        validated_logs = list(log_collection.find(log_query))
        df_logs = pd.DataFrame(validated_logs)
        user_logs = df_logs[df_logs["username"] == current_user]
        user_validated_pids = user_logs["pid"].nunique()

        total_matching = len(matching_properties)
        validation_rate = (user_validated_pids / total_matching * 100) if total_matching > 0 else 0

        # --- Display Metrics ---
        if not unvalidated_docs:
            st.warning("No unvalidated properties found for the selected filters.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Validated by {current_user}", user_validated_pids)
            col2.metric(label="Unvalidated Properties", value=len(unvalidated_docs))
            col3.metric(label="Your Validation Rate (%)", value=f"{validation_rate:.2f}%")

        # --- Validation Rate Over Time ---
        st.subheader("Validation Activity Over Time")
        if not user_logs.empty:
            user_logs["date"] = pd.to_datetime(user_logs["update_timestamp"]).dt.date
            trend_df = user_logs.groupby("date").size().reset_index(name="Validations")
            fig_trend = px.line(trend_df, x="date", y="Validations", markers=True)
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Properties Validated")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No validation data found for the current user.")

    # --------------------- Users Validation Info ---------------------
    st.markdown(f"<h1 style='text-align:center;'>Users Validation Info</h1>", unsafe_allow_html=True)

    # --- Session defaults ---
    if "log_filter_date_range" not in st.session_state:
        st.session_state.log_filter_date_range = [datetime.today() - timedelta(days=30), datetime.today()]
    if "log_filter_source" not in st.session_state:
        st.session_state.log_filter_source = "All"
    if "log_filter_ai_type" not in st.session_state:
        st.session_state.log_filter_ai_type = "All"
    if "log_filter_user" not in st.session_state:
        st.session_state.log_filter_user = "All"

    # --- UI Filters ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.date_input(
            "Select Date Range",
            help="Select the date range for the logs that are validated",
            value=st.session_state.log_filter_date_range,
            key="log_filter_date_range"
        )
    with col2:
        st.selectbox(
            "Select Source",
            options=["All", "rightmove", "zoopla"],
            index=(["All", "rightmove", "zoopla"]).index(st.session_state.log_filter_source),
            key="log_filter_source"
        )
    with col3:
        st.selectbox(
            "Select AI Type",
            options=["All", "Potential", "Under Value"],
            index=(["All", "Potential", "Under Value"]).index(st.session_state.log_filter_ai_type),
            key="log_filter_ai_type"
        )
    with col4:
        selected_source = st.session_state.log_filter_source
        selected_ai_type = st.session_state.log_filter_ai_type

        # Dynamically load available users from relevant collections
        if selected_source == "zoopla":
            if selected_ai_type == "Potential":
                logs = list(client["zoopla_data"]["zoopla_ai_update_logs"].find({"update_timestamp": {"$exists": True}}))
            elif selected_ai_type == "Under Value":
                logs = list(client["zoopla_data"]["zoopla_uv_update_logs"].find({"update_timestamp": {"$exists": True}}))
            else:
                logs = list(client["zoopla_data"]["zoopla_ai_update_logs"].find({"update_timestamp": {"$exists": True}})) + \
                       list(client["zoopla_data"]["zoopla_uv_update_logs"].find({"update_timestamp": {"$exists": True}}))
        elif selected_source == "rightmove":
            if selected_ai_type == "Potential":
                logs = list(client["rightmove_data"]["rightmove_ai_update_logs"].find({"update_timestamp": {"$exists": True}}))
            elif selected_ai_type == "Under Value":
                logs = list(client["rightmove_data"]["rightmove_uv_update_logs"].find({"update_timestamp": {"$exists": True}}))
            else:
                logs = list(client["rightmove_data"]["rightmove_ai_update_logs"].find({"update_timestamp": {"$exists": True}})) + \
                       list(client["rightmove_data"]["rightmove_uv_update_logs"].find({"update_timestamp": {"$exists": True}}))
        else:
            # All sources, all AI types
            logs = list(client["rightmove_data"]["rightmove_ai_update_logs"].find({"update_timestamp": {"$exists": True}})) + \
                   list(client["rightmove_data"]["rightmove_uv_update_logs"].find({"update_timestamp": {"$exists": True}})) + \
                   list(client["zoopla_data"]["zoopla_ai_update_logs"].find({"update_timestamp": {"$exists": True}})) + \
                   list(client["zoopla_data"]["zoopla_uv_update_logs"].find({"update_timestamp": {"$exists": True}}))

        df_logs = pd.DataFrame(logs)
        df_logs["update_timestamp"] = pd.to_datetime(df_logs["update_timestamp"])

        available_users = df_logs["username"].dropna().unique().tolist()
        st.selectbox(
            "Select User",
            options=["All"] + available_users,
            index=(["All"] + available_users).index(st.session_state.get("log_filter_user", "All")),
            key="log_filter_user"
        )

    # --- Apply Filters ---
    if st.button("Apply Filters"):
        filtered_df = df_logs.copy()
        start_date, end_date = pd.to_datetime(st.session_state.log_filter_date_range[0]), pd.to_datetime(st.session_state.log_filter_date_range[1])
        filtered_df = filtered_df[
            (filtered_df["update_timestamp"] >= start_date) &
            (filtered_df["update_timestamp"] <= end_date)
        ]

        if st.session_state.log_filter_source != "All":
            filtered_df = filtered_df[filtered_df["source"] == st.session_state.log_filter_source]
        if st.session_state.log_filter_ai_type != "All":
            filtered_df = filtered_df[filtered_df["ai_type"] == st.session_state.log_filter_ai_type]
        if st.session_state.log_filter_user != "All":
            filtered_df = filtered_df[filtered_df["username"] == st.session_state.log_filter_user]

        if not filtered_df.empty:
            display_cols = ["username", "pid", "ai_type", "source", "convertible_status", "value_prediction", "update_timestamp"]
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            filtered_df_display = filtered_df[available_cols].sort_values(by="update_timestamp", ascending=False)

            st.success(f"Found {len(filtered_df_display)} validated properties.")
            st.dataframe(filtered_df_display, use_container_width=True)
        else:
            st.warning("No validated properties found for the selected filters.")
    
# --------------------- VALIDATION DASHBOARD ---------------------
elif page == "Validation Dashboard":

    st.markdown(f"<h1 style='text-align:center;'>Validation Dashboard</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        date_range = st.date_input("Date Range", [datetime.today() - timedelta(days=30), datetime.today()], key="dash_date_range")
    with col2:
        source = st.selectbox("Source", ["All", "rightmove", "zoopla"], key="dash_source")
    with col3:
        convertible_status = st.selectbox("Convertible Status", ["All", 1, 2, 3, 4], key="dash_status")
    with col4:
        property_type = st.selectbox("Property Type", ["All", "filter excluded", "flat", "house", "hmo"], key="dash_prop_type")
    with col5:
        ai_type = st.selectbox("AI Type", ["All", "Potential", "Under Value"], key="dash_ai_type")

    submitted = st.button("Apply Dashboard Filters")


    if submitted:
        start_date = datetime.combine(st.session_state.date_from, datetime.min.time())
        end_date = datetime.combine(st.session_state.date_to, datetime.max.time())

        query = {
            "etl_load_timestamp": {"$gte": start_date, "$lte": end_date},
            "update_timestamp": {"$exists": False}
        }

        # ---------- Database & Collection Mapping ----------
        db_name = None
        log_collection = None
        potential_collection = None
        undervalue_collection = None

        if source == "rightmove" and ai_type == "Potential":
            db_name = "rightmove_data"
            log_collection = client[db_name]["rightmove_ai_update_logs"]
            collection_to_query = client[db_name]["rightmove_outcode_ai"]

        elif source == "rightmove" and ai_type == "Under Value":
            db_name = "rightmove_data"
            log_collection = client[db_name]["rightmove_uv_update_logs"]
            collection_to_query = client[db_name]["rightmove_outcode_uv"]

        elif source == "zoopla" and ai_type == "Potential":
            db_name = "zoopla_data"
            log_collection = client[db_name]["zoopla_ai_update_logs"]
            collection_to_query = client[db_name]["zoopla_outcode_ai"]

        elif source == "zoopla" and ai_type == "Under Value":
            db_name = "zoopla_data"
            log_collection = client[db_name]["zoopla_uv_update_logs"]
            collection_to_query = client[db_name]["zoopla_outcode_uv"]

        else:
            # Source = All or AI = All → Use both
            potential_collection = client["rightmove_data"]["rightmove_outcode_ai"] if source in ["All", "rightmove"] else client["zoopla_data"]["zoopla_outcode_ai"]
            undervalue_collection = client["rightmove_data"]["rightmove_outcode_uv"] if source in ["All", "rightmove"] else client["zoopla_data"]["zoopla_outcode_uv"]
            collection_to_query = None  # Both will be queried below

        if source != "All":
            query["source"] = source
        if convertible_status != "All":
            query["convertible_status"] = int(convertible_status)
        if property_type != "All":
            query["filter_property_type"] = property_type

        # ---------------- Query Based on Collection ----------------
        if collection_to_query is not None:
            data = list(collection_to_query.find(query))
        else:
            data = list(potential_collection.find(query)) + list(undervalue_collection.find(query))

        if not data:
            st.warning("No data found for the given filters.")
            st.stop()

        df = pd.DataFrame(data)

        df["etl_load_timestamp"] = pd.to_datetime(df["etl_load_timestamp"], errors="coerce")
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
                    labels={"filter_property_type": "Property Type"}
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Bedroom Distribution")
            fig = px.box(
                df,
                y="bedroom",
                labels={"bedroom": "Bedrooms"}
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Listings Over Time")
            if not df.empty:
                time_series = df.groupby(df["etl_load_timestamp"].dt.date).size().reset_index(name="Count")
                time_series["etl_load_timestamp"] = pd.to_datetime(time_series["etl_load_timestamp"])
                fig = px.line(
                    time_series,
                    x="etl_load_timestamp",
                    y="Count",
                    title="Listings by Load Date",
                    labels={"etl_load_timestamp": "Date", "Count": "Listings"}
                )
                st.plotly_chart(fig, use_container_width=True)
        with col4:
            if "postcode" in df.columns and "bedroom" in df.columns:
                st.subheader("Postcode vs Bedroom Heatmap")
                heatmap_data = pd.crosstab(df["postcode"], df["bedroom"].astype(int))
                st.dataframe(heatmap_data.style.background_gradient(cmap="Blues"))

        col5, col6 = st.columns(2)
        with col5:   
            st.subheader("Validation Convertible Status")     
            def get_status_counts_by_date(query):
                if collection_to_query is not None:
                    if ai_type == "Potential":
                        docs = collection_to_query.find(query, {"convertible_status": 1})
                        field = "convertible_status"
                    elif ai_type == "Under Value":
                        docs = collection_to_query.find(query, {"value_prediction": 1})
                        field = "value_prediction"
                    else:
                        field = None
                        docs = []
                else:
                    docs = list(potential_collection.find(query, {"convertible_status": 1})) + \
                        list(undervalue_collection.find(query, {"value_prediction": 1}))
                    field = None

                counts = {}
                for doc in docs:
                    if field:
                        value = doc.get(field)
                    else:
                        value = doc.get("convertible_status") or doc.get("value_prediction")
                    if value is not None:
                        counts[value] = counts.get(value, 0) + 1
                return dict(sorted(counts.items()))
            status_query = {
                "etl_load_timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "update_timestamp": {"$exists": False}
            }

            counts = get_status_counts_by_date(status_query)

            if not counts:
                st.warning("No convertible status data found for the selected date range.")
            else:
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

# --------------------- ENGINE DASHBOARD ---------------------
elif page == "Engine Dashboard":
    render_engine_dashboard()

--------------------- MONTHLY TASK DASHBOARD ---------------------
elif page == "Monthly Task":  
    render_monthly_task()
