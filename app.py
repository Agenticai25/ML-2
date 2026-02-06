import streamlit as st
import pandas as pd
import json
import requests
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

# --- Initialize Session State ---
if "data" not in st.session_state: st.session_state.data = None
if "built_payload" not in st.session_state: st.session_state.built_payload = None
if "inference_result" not in st.session_state: st.session_state.inference_result = None
if "final_display_df" not in st.session_state: st.session_state.final_display_df = None

# --- File Upload Logic ---
uploaded_file = st.file_uploader("Upload inference CSV/Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success(f"File loaded: {len(df)} records found.")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Prediction Logic ---
if st.session_state.data is not None:
    row_num = st.number_input("Select Row for Prediction", min_value=1, max_value=len(st.session_state.data), value=1)

    if st.button("Build json payload"):
        row_data = st.session_state.data.iloc[row_num - 1].to_dict()
        row_data_cleaned = json.loads(pd.Series(row_data).to_json(date_format='iso'))
        st.session_state.built_payload = json.dumps(row_data_cleaned, indent=2)
        st.code(st.session_state.built_payload, language="json")

    if st.button("Inference", type="primary", disabled=not st.session_state.built_payload):
        try:
            payload_json = json.loads(st.session_state.built_payload)
            with st.spinner(f"Sending request to Azure..."):
                response = requests.post(
                    api_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout_val,
                    verify=False
                )

                if response.status_code == 200:
                    st.session_state.inference_result = response.json()
                    st.success("Success!")

                    results_list = st.session_state.inference_result.get("results", [])
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        selected_extra_columns = ["msdyn_receiveddate"]

                        extra_data = st.session_state.data.iloc[[row_num - 1]][selected_extra_columns].reset_index(
                            drop=True)

                        if 'predicted_resolution_minutes' in results_df.columns:
                            results_df['predicted_resolution_minutes'] = pd.to_numeric(
                                results_df['predicted_resolution_minutes']).round(0).astype(int)

                        # Store unformatted datetimes for delta calculation
                        results_df['_dt_pred'] = pd.to_datetime(results_df['predicted_resolved_date'])

                        # Formatting for display
                        extra_data['msdyn_receiveddate'] = pd.to_datetime(extra_data['msdyn_receiveddate']).dt.strftime(
                            '%m/%d/%Y %H:%M:%S')
                        results_df['predicted_resolved_date'] = results_df['_dt_pred'].dt.strftime('%m/%d/%Y %H:%M:%S')

                        # Combine into session state
                        st.session_state.final_display_df = pd.concat([results_df, extra_data], axis=1)
                    else:
                        st.warning("No results found in the response.")
                else:
                    st.error(f"Error: Server returned {response.status_code}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # --- Integrated Result Section ---
    if st.session_state.final_display_df is not None:
        st.markdown("---")
        st.subheader("Prediction Analysis & Validation with Actuals")

        # Second File Uploader
        actuals_file = st.file_uploader("Upload Actuals file to calculate Delta", type=["csv", "xlsx"])

        if actuals_file:
            try:
                act_df = pd.read_csv(actuals_file) if actuals_file.name.endswith('.csv') else pd.read_excel(
                    actuals_file)
                tid = st.session_state.final_display_df["TicketNumber"].iloc[0]
                match = act_df[act_df["TicketNumber"] == tid]

                if not match.empty:
                    # Extraction & Calculation
                    actual_dt = pd.to_datetime(match["actual_resolve_dt"].iloc[0])
                    pred_dt = st.session_state.final_display_df["_dt_pred"].iloc[0]
                    delta_mins = ((actual_dt - pred_dt).total_seconds() / 60)
                    actual_dur = match["actual_Duration"].iloc[0]

                    # Update existing session state dataframe
                    st.session_state.final_display_df["Actual_Resolve_Value"] = actual_dt.strftime('%m/%d/%Y %H:%M:%S')
                    st.session_state.final_display_df["actual_Duration"] = actual_dur
                    st.session_state.final_display_df["Delta_Minutes"] = abs(round(delta_mins, 0))
                    st.session_state.final_display_df["SLA_Status"] = "Early" if delta_mins > 0 else "Delay"

                    st.success("Actuals and Delta updated successfully!")
                else:
                    st.warning(f"Ticket {tid} not found in actuals file.")
            except Exception as e:
                st.error(f"Error matching actuals: {e}")

        # --- Display Configuration ---
        column_order = [
            "TicketNumber",
            "msdyn_receiveddate",
            "predicted_resolved_date",
            "Actual_Resolve_Value",
            "actual_Duration",
            "predicted_resolution_minutes",
            "Delta_Minutes",
            "SLA_Status"
        ]

        column_mapping = {
            "TicketNumber": "Ticket ID",
            "msdyn_receiveddate": "Received Date",
            "predicted_resolved_date": "Predicted Resolution Date",
            "Actual_Resolve_Value": "Actual Resolution Date",
            "actual_Duration": "Actual Duration (Mins)",
            "predicted_resolution_minutes": "Predicted Duration (Mins)",
            "Delta_Minutes": "Delta (Mins)",
            "SLA_Status": "Status"
        }

        # Filter to show only columns that exist
        cols_to_show = [c for c in column_order if c in st.session_state.final_display_df.columns]

        # Prepare display dataframe with renamed columns
        display_df = st.session_state.final_display_df[cols_to_show].rename(columns=column_mapping)

        # Final Table Output (Standard display, no background highlighting)
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("Please upload a file to begin.")
