import streamlit as st
import pandas as pd
import json
import requests
from io import BytesIO
import urllib3
import base64
import os

# Suppress the "InsecureRequestWarning"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- Helper Function to Encode Local Image ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None
    except Exception:
        return None


# --- Page Config ---
st.set_page_config(
    page_title="Cosmic Case SLA Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the Image ---
img_path = os.path.join("assets", "microsoft-logo.png")
img_base64 = get_base64_image(img_path)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; background-color: #faf9f8; color: #201f1e; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1dfdd; }
    h1 { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #201f1e; font-size: 1.5rem; margin-top: 0px; }
    .stButton > button { background-color: #ffffff; color: #323130; border: 1px solid #8a8886; border-radius: 2px; font-size: 14px; }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] { background-color: #0078d4; color: white; border: none; }
    .header-logo { height: 50px; width: auto; margin-right: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Connection")
    api_url = st.text_input("Inference Endpoint",
                            value="https://func-sla-catboost-infer-uat-eastus.azurewebsites.net/predict")

    st.markdown("---")
    st.markdown("### Request options")
    timeout_val = st.number_input("Connection Timeout (seconds)", value=240)

# --- Main Content Header ---
col1, col2 = st.columns([0.08, 0.92])
with col1:
    if img_base64:
        st.markdown(f'<img src="data:image/png;base64,{img_base64}" class="header-logo">', unsafe_allow_html=True)
    else:
        st.write("🏢")
with col2:
    st.markdown("# Cosmic Case SLA Prediction with CatBoost")

# --- File Upload Logic ---
uploaded_file = st.file_uploader("Upload inference CSV/Excel", type=["csv", "xlsx", "xls"])

if "data" not in st.session_state: st.session_state.data = None
if "built_payload" not in st.session_state: st.session_state.built_payload = None
if "inference_result" not in st.session_state: st.session_state.inference_result = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success(f"File loaded: {len(df)} records found.")
        st.dataframe(df, use_container_width=True)

        # with st.expander("View Data Column Types"):
        #     st.write(df.dtypes.to_dict())

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Prediction Logic ---
if st.session_state.data is not None:
    row_num = st.number_input("Select Row for Prediction", min_value=1, max_value=len(st.session_state.data), value=1)

    if st.button("Build json payload"):
        row_data = st.session_state.data.iloc[row_num - 1].to_dict()
        # Clean up types for JSON serialization
        row_data_cleaned = json.loads(pd.Series(row_data).to_json(date_format='iso'))
        st.session_state.built_payload = json.dumps(row_data_cleaned, indent=2)
        st.code(st.session_state.built_payload, language="json")

    if st.button("Inference", type="primary", disabled=not st.session_state.built_payload):
        try:
            payload_json = json.loads(st.session_state.built_payload)

            with st.spinner(f"Sending request to Azure..."):
                # DEBUG: Print payload to console
                print(f"DEBUG SENDING: {payload_json}")

                response = requests.post(
                    api_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout_val,
                    verify=False
                )

                # DEBUG: Show what happened in the UI
                # with st.expander("🪲 Debug Information", expanded=True):
                #     st.write("**Endpoint:**", api_url)
                #     st.write("**Payload Sent:**")
                #     st.json(payload_json)
                #     st.write("**Response Status Code:**", response.status_code)

                if response.status_code == 200:
                    st.session_state.inference_result = response.json()
                    st.success("Success!")
                    st.write("**Prediction Output:**")
                    # st.json(st.session_state.inference_result)
                    # --- REVISED SECTION START ---
                    # Access the 'results' list from the JSON and display as a table
                    results_list = st.session_state.inference_result.get("results", [])
                    if results_list:
                        # 1. Convert the Azure results to a DataFrame
                        results_df = pd.DataFrame(results_list)

                        # 2. DEFINE EXTRA COLUMNS: Add your desired column names from the data file here
                        selected_extra_columns = ["msdyn_receiveddate","actual_resolve_dt"]

                        # 3. Extract those columns from the specific row we just predicted
                        # We use [row_num - 1] to match the row used for the "Build payload" step
                        extra_data = st.session_state.data.iloc[[row_num - 1]][selected_extra_columns].reset_index(
                            drop=True)

                        ##Remove decimals from resolution minutes
                        if 'predicted_resolution_minutes' in results_df.columns:
                            results_df['predicted_resolution_minutes'] = pd.to_numeric(
                                results_df['predicted_resolution_minutes']).round(0).astype(int)

                        ##ensure both are same datetime objects
                        extra_data['msdyn_receiveddate'] = pd.to_datetime(extra_data['msdyn_receiveddate'])
                        results_df['predicted_resolved_date'] = pd.to_datetime(results_df['predicted_resolved_date'])

                        extra_data['msdyn_receiveddate'] = extra_data['msdyn_receiveddate'].dt.strftime(
                            '%m/%d/%Y %H:%M:%S')
                        results_df['predicted_resolved_date'] = results_df['predicted_resolved_date'].dt.strftime(
                            '%m/%d/%Y %H:%M:%S')

                        # 4. Combine the extra columns with the results
                        final_display_df = pd.concat([results_df, extra_data], axis=1)

                        ##List your columns in the exact order you want them to appear
                        column_order = [
                            "TicketNumber",
                            "msdyn_receiveddate",
                            "predicted_resolved_date",
                            "actual_resolve_dt",
                            "predicted_resolution_minutes"
                        ]
                        ###Apply the order
                        existing_cols = [col for col in column_order if col in final_display_df.columns]
                        final_display_df = final_display_df[existing_cols]

                        # 5. Display the final table
                        st.dataframe(final_display_df, use_container_width=True)
                    else:
                        st.warning("No results found in the response.")                    # --- REVISED SECTION END ---
                else:
                    st.error(f"Error: Server returned {response.status_code}")
                    st.write(response.text)

        except requests.exceptions.Timeout:
            st.error("The request timed out.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload a file to begin.")