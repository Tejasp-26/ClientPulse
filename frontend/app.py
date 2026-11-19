# frontend/app.py
# Simple Streamlit UI for ClientPulse.
# It only talks to the FastAPI backend using the requests library.

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ClientPulse", layout="wide")


# ---------- helper functions ----------
def money(x):
    return f"Rs {x:,.0f}"


def get(path, params=None):
    try:
        r = requests.get(API_URL + path, params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Start it with: uvicorn backend.main:app --reload"
    except Exception as e:
        return None, str(e)


def post(path, json_data=None, files=None, params=None):
    try:
        r = requests.post(API_URL + path, json=json_data, files=files, params=params, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Start it with: uvicorn backend.main:app --reload"
    except Exception as e:
        return None, str(e)


# ---------- sidebar ----------
st.sidebar.title("ClientPulse")
st.sidebar.write("Client churn predictor")

hist, err = get("/history/summary")
if err:
    st.sidebar.error(err)
elif hist and hist.get("clients", 0) > 0:
    st.sidebar.write(f"Clients in database: {hist['clients']}")
    st.sidebar.write(f"Months: {hist['first_month']} to {hist['last_month']}")
else:
    st.sidebar.write("No data in database yet.")

st.sidebar.subheader("Add data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file and st.sidebar.button("Upload"):
    result, err = post("/upload", files={"file": (uploaded_file.name, uploaded_file.read(), "text/csv")})
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Saved {result['rows_saved']} rows for {result['clients']} clients")
        st.rerun()

if st.sidebar.button("Load sample data"):
    result, err = post("/load-sample")
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Loaded {result['rows_saved']} rows")
        st.rerun()

st.sidebar.subheader("Risk levels")
high_pct = st.sidebar.slider("High risk from (%)", 50, 99, 70)
medium_pct = st.sidebar.slider("Medium risk from (%)", 10, 69, 40)
high_thresh = high_pct / 100
medium_thresh = medium_pct / 100

if st.sidebar.button("Run analysis"):
    result, err = post("/analyse", json_data={"high": high_thresh, "medium": medium_thresh})
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(
            f"High: {result['high_risk']}, Medium: {result['medium_risk']}, Low: {result['low_risk']}"
        )
        st.rerun()

template_cols = [
    "client_id", "month", "logins", "features_used", "tickets",
    "last_login_days_ago", "contract_end_date", "plan_type", "monthly_revenue",
]
st.sidebar.download_button(
    "Download template CSV",
    pd.DataFrame(columns=template_cols).to_csv(index=False),
    file_name="clientpulse_template.csv",
    mime="text/csv",
)

# ---------- load main data ----------
overview, ov_err = get("/overview", params={"high": high_thresh, "medium": medium_thresh})
run_id = overview["run_id"] if overview else None

st.title("ClientPulse")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Client list", "Client details", "Model"])


# ---------- tab 1: overview ----------
with tab1:
    if ov_err or not overview:
        st.write("Load the sample data and click Run analysis (in the sidebar) to start.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clients analysed", overview["clients_analysed"])
        c2.metric("Predicted churn rate", f"{overview['predicted_churn_rate']:.1%}")
        c3.metric("High risk clients", overview["high_risk"])
        c4.metric("Monthly revenue at risk", money(overview.get("monthly_revenue_at_risk", 0)))

        st.write(f"Most common warning sign: {overview.get('top_warning_sign', 'N/A')}")

        left, right = st.columns(2)
        with left:
            st.subheader("Clients by risk level")
            groups = overview.get("risk_group_counts", {})
            if groups:
                st.bar_chart(pd.Series(groups))
        with right:
            st.subheader("Revenue at risk by plan")
            by_plan = overview.get("revenue_at_risk_by_plan", {})
            if by_plan:
                st.bar_chart(pd.Series(by_plan))
            else:
                st.write("No high risk clients.")

        st.subheader("Common warning signs")
        reasons = overview.get("top_reasons_counts", {})
        if reasons:
            st.bar_chart(pd.Series(reasons).sort_values(ascending=False).head(6))

        st.subheader("Estimated savings")
        a, b = st.columns(2)
        contact_pct = a.slider("Percent of clients to contact", 5, 50, 20, 5)
        save_pct = b.slider("Percent you expect to retain", 10, 80, 30, 5)
        impact, imp_err = get("/impact", params={"contact_pct": contact_pct, "save_pct": save_pct})
        if impact:
            i1, i2, i3 = st.columns(3)
            i1.metric("Clients contacted", impact["clients_contacted"])
            i2.metric("Likely leavers", impact["likely_leavers"])
            i3.metric("Saved per year", money(impact["saved_per_year"]))
            st.caption("This is only an estimate based on the values you chose.")

        st.subheader("Clients whose risk increased since last run")
        changes, ch_err = get("/changes")
        if changes and changes.get("has_previous"):
            risers = changes.get("risers", [])
            if risers:
                df = pd.DataFrame(risers)
                df.columns = ["Client", "Previous risk", "Current risk", "Change"]
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.write("No client got much worse.")
        else:
            st.write("Run the analysis again later to see changes.")


# ---------- tab 2: client list ----------
with tab2:
    if not run_id:
        st.write("Run the analysis first.")
    else:
        levels = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        plans = st.multiselect("Plan", ["Standard", "Premium", "Enterprise"],
                               default=["Standard", "Premium", "Enterprise"])

        data, err = get("/clients", params={"run_id": run_id})
        if err:
            st.error(err)
        elif data:
            df = pd.DataFrame(data)
            df = df[df["level"].isin(levels) & df["plan"].isin(plans)]
            table = pd.DataFrame({
                "Client": df["client_id"],
                "Risk %": (df["risk"] * 100).round(1),
                "Level": df["level"],
                "Plan": df["plan"],
                "Monthly revenue": df["monthly_revenue"].apply(money),
            })
            st.dataframe(table, hide_index=True, use_container_width=True)

            export = requests.get(API_URL + "/clients/export", params={"run_id": run_id}, timeout=30)
            if export.ok:
                st.download_button("Download CSV", export.content,
                                   file_name=f"clients_{run_id}.csv", mime="text/csv")


# ---------- tab 3: client details ----------
with tab3:
    if not run_id:
        st.write("Run the analysis first.")
    else:
        data, err = get("/clients", params={"run_id": run_id})
        if data:
            client_id = st.selectbox("Choose a client", [c["client_id"] for c in data])
            d, err = get(f"/clients/{client_id}", params={"run_id": run_id})

            if err:
                st.error(err)
            elif d:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Risk", f"{d['risk']:.0%}", d["risk_level"], delta_color="off")
                prev = d.get("previous_risk")
                c2.metric("Previous risk", f"{prev:.0%}" if prev is not None else "None")
                c3.metric("Plan", d["plan_type"])
                c4.metric("Monthly revenue", money(d["monthly_revenue"]))

                left, right = st.columns(2)
                with left:
                    st.subheader("Logins per month")
                    trend = d.get("login_trend", [])
                    if trend:
                        trend_df = pd.DataFrame(trend).set_index("month")
                        st.line_chart(trend_df["logins"])
                with right:
                    st.subheader("What affected the risk")
                    shap_vals = d.get("shap_values", [])
                    if shap_vals:
                        shap_df = pd.DataFrame(shap_vals).set_index("friendly_name")
                        st.bar_chart(shap_df["value"])
                        st.caption("Positive value raises the risk, negative value lowers it.")

                left, right = st.columns(2)
                with left:
                    st.subheader("Warning signs")
                    for r in d.get("reasons", []):
                        st.write("- " + r)
                with right:
                    st.subheader("Suggested actions")
                    for a in d.get("actions", []):
                        st.write("- " + a)

                st.subheader("AI note")
                if st.button("Generate note"):
                    note, err = post(f"/clients/{client_id}/note", params={"run_id": run_id})
                    if err:
                        st.error(err)
                    else:
                        st.write(note["note"])
                        st.caption("Source: " + note["source"])

                st.subheader("Feedback")
                note_text = st.text_input("Note (optional)")
                options = {
                    "Good call": "good_call",
                    "False alarm": "false_alarm",
                    "Client left": "left",
                    "Client stayed": "stayed",
                }
                cols = st.columns(4)
                for col, (label, outcome) in zip(cols, options.items()):
                    if col.button(label):
                        res, err = post("/feedback", json_data={
                            "run_id": run_id,
                            "client_id": client_id,
                            "outcome": outcome,
                            "note": note_text,
                        })
                        if err:
                            st.error(err)
                        else:
                            st.success("Feedback saved")

                past = d.get("feedback", [])
                if past:
                    st.write("Past feedback:")
                    st.dataframe(pd.DataFrame(past)[["outcome", "note", "created_at"]],
                                 hide_index=True, use_container_width=True)


# ---------- tab 4: model ----------
with tab4:
    st.write("Note: the model is trained on synthetic data, so these scores show that the "
             "pipeline works, not real-world accuracy. Risk % is a ranking score, not a probability.")

    info, err = get("/model/info")
    if err:
        st.error(err)
    elif info:
        metrics = info.get("metrics", {})
        left, right = st.columns(2)
        with left:
            st.subheader("XGBoost")
            for k, v in metrics.get("xgboost", {}).items():
                if k != "model":
                    st.write(f"{k.replace('_', ' ')}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        with right:
            st.subheader("Logistic Regression")
            for k, v in metrics.get("logistic_regression", {}).items():
                if k != "model":
                    st.write(f"{k.replace('_', ' ')}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        st.subheader("Features used")
        st.write(", ".join(info.get("feature_columns", [])))

    st.subheader("Feedback summary")
    fb, err = get("/feedback/summary")
    if fb:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", fb.get("total_feedback", 0))
        c2.metric("Good calls", fb.get("good_call", 0))
        c3.metric("False alarms", fb.get("false_alarm", 0))
        rate = fb.get("good_call_rate")
        c4.metric("Good call rate", f"{rate:.0%}" if rate is not None else "N/A")