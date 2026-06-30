"""Streamlit demo app for the EyeV A&G reasoning engine."""

from datetime import datetime
import json
from pathlib import Path
from urllib.request import Request, urlopen

import streamlit as st

from EyeV_OKG_V7_engine import OKGEngine


APP_DIR = Path(__file__).resolve().parent
GRAPH_FILE = APP_DIR / "EyeV_Ophthalmic_Knowledge_Graph_v2.xlsx"


EXAMPLES = [
    "Patient has new flashes and floaters but no curtain or field loss.",
    "Patient has new flashes and floaters with a curtain coming across vision.",
    "Blurred central vision with distortion and OCT showing subretinal fluid.",
    "Raised IOP 28mmHg with suspicious optic discs but visual fields not supplied.",
    "Painful red eye with photophobia and anterior chamber cells.",
    "Post cataract OCT shows cystic spaces and possible subtle post op CMO, vision unaffected and no symptoms of uveitis.",
    "Indistinct optic nerve margins with severe headache and vomiting, transient visual obscurations and possible swollen disc.",
    "Known optic disc drusen and pseudopapilloedema stable with no symptoms and no new referral indicated.",
    "Pain on eye movements with reduced colour vision and central visual field defect, no red eye.",
    "Referral ID cannot be found and patient has not heard anything about triage.",
]

SHEET_COLUMNS = [
    "Timestamp",
    "Question",
    "Outcome ID",
    "Outcome",
    "Outcome rationale",
    "Top presentation ID",
    "Top presentation",
    "Top presentation confidence",
    "Safety IDs",
    "Safety conditions",
    "Detected feature IDs",
    "Detected features",
    "Missing information IDs",
    "Missing information",
    "Draft summary",
    "Draft suggested response",
    "Draft safety net",
]


st.set_page_config(
    page_title="EyeV A&G Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_password():
    """Optional password gate for public demo links."""
    demo_password = st.secrets.get("APP_PASSWORD", "")
    if not demo_password:
        return True

    if st.session_state.get("password_ok"):
        return True

    st.title("EyeV A&G Tool")
    password = st.text_input("Password", type="password")
    if st.button("Enter"):
        if password == demo_password:
            st.session_state["password_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


@st.cache_resource
def load_engine():
    return OKGEngine(str(GRAPH_FILE))


def logging_mode():
    if st.secrets.get("GOOGLE_APPS_SCRIPT_URL", ""):
        return "apps_script"
    if st.secrets.get("GOOGLE_SHEET_ID", "") and st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None):
        return "service_account"
    return "not_configured"


@st.cache_resource
def load_google_sheet():
    sheet_id = st.secrets.get("GOOGLE_SHEET_ID", "")
    worksheet_name = st.secrets.get("GOOGLE_WORKSHEET_NAME", "A&G Log")
    service_account_info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)

    if not sheet_id or not service_account_info:
        return None

    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(
        dict(service_account_info),
        scopes=scopes,
    )
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=len(SHEET_COLUMNS))

    values = worksheet.row_values(1)
    if values != SHEET_COLUMNS:
        worksheet.update("A1:Q1", [SHEET_COLUMNS])

    return worksheet


def append_with_apps_script(row):
    url = st.secrets.get("GOOGLE_APPS_SCRIPT_URL", "")
    if not url:
        return False

    payload = {
        "token": st.secrets.get("GOOGLE_LOG_TOKEN", ""),
        "headers": SHEET_COLUMNS,
        "row": row,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(body.get("error", "Google Apps Script logging failed"))
    return True


def join_values(items, key):
    return "; ".join(str(item.get(key, "")) for item in items if item.get(key, ""))


def result_to_sheet_row(question, result):
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_presentation = presentations[0] if presentations else {}
    draft = result.get("Draft Response", {})

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        outcome.get("Rationale", ""),
        top_presentation.get("Presentation ID", ""),
        top_presentation.get("Presentation", ""),
        top_presentation.get("Confidence", ""),
        join_values(result.get("Safety Ranking", []), "Safety Condition ID"),
        join_values(result.get("Safety Ranking", []), "Safety Condition"),
        join_values(result.get("Detected Features", []), "Feature ID"),
        join_values(result.get("Detected Features", []), "Feature"),
        join_values(result.get("Missing Information", []), "Missing Information ID"),
        join_values(result.get("Missing Information", []), "Missing Information"),
        draft.get("Summary", ""),
        draft.get("Suggested response", ""),
        draft.get("Safety net", ""),
    ]


def log_to_google_sheet(question, result):
    row = result_to_sheet_row(question, result)

    if st.secrets.get("GOOGLE_APPS_SCRIPT_URL", ""):
        append_with_apps_script(row)
        return "logged"

    worksheet = load_google_sheet()
    if worksheet is None:
        return "not_configured"

    worksheet.append_row(
        row,
        value_input_option="USER_ENTERED",
    )
    return "logged"


def outcome_colour(outcome_id):
    if outcome_id == "OUT003":
        return "#9b1c31"
    if outcome_id == "OUT002":
        return "#a15c00"
    return "#286140"


def render_outcome(outcome):
    outcome_id = outcome.get("Outcome ID", "-")
    colour = outcome_colour(outcome_id)
    st.markdown(
        f"""
        <div style="border-left: 6px solid {colour}; padding: 0.9rem 1rem; background: #ffffff; border-radius: 0.4rem; border-top: 1px solid #d9e0e7; border-right: 1px solid #d9e0e7; border-bottom: 1px solid #d9e0e7;">
          <div style="font-weight: 800; font-size: 1.05rem;">{outcome_id}: {outcome.get("Outcome", "-")}</div>
          <div style="color: #5c6976; margin-top: 0.25rem;">{outcome.get("Rationale", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_draft_response(draft):
    if not draft:
        return

    with st.container(border=True):
        st.markdown("**Summary**")
        st.write(draft.get("Summary", ""))
        st.markdown("**Suggested response**")
        st.write(draft.get("Suggested response", ""))
        st.markdown("**Safety net**")
        st.write(draft.get("Safety net", ""))


def render_features(features):
    if not features:
        st.info("No clinical features detected.")
        return

    for feature in features:
        with st.container(border=True):
            st.markdown(f"**{feature['Feature ID']}**  {feature['Feature']}")
            terms = feature.get("Matched Terms", [])
            if terms:
                st.caption("Matched: " + ", ".join(str(term) for term in terms))


def render_presentations(presentations):
    if not presentations:
        st.info("No confident presentation identified.")
        return

    for presentation in presentations:
        with st.container(border=True):
            left, right = st.columns([0.75, 0.25])
            left.markdown(f"**{presentation['Presentation ID']}**  {presentation['Presentation']}")
            right.metric("Confidence", f"{presentation['Confidence']}%")

            evidence = presentation.get("Evidence", [])
            if evidence:
                with st.expander("Evidence"):
                    for item in evidence:
                        st.write(
                            f"{item['Source Entity ID']} - {item['Source Entity']} | weight {item['Weight']}"
                        )


def render_safety(safety_items):
    if not safety_items:
        st.info("No safety condition triggered.")
        return

    for safety in safety_items:
        with st.container(border=True):
            left, right = st.columns([0.75, 0.25])
            left.markdown(f"**{safety['Safety Condition ID']}**  {safety['Safety Condition']}")
            right.metric("Confidence", f"{safety['Confidence']}%")


def render_missing_info(items):
    if not items:
        st.info("No missing information requested.")
        return

    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item['Missing Information ID']}**  {item['Missing Information']}")


def main():
    if not check_password():
        return

    engine = load_engine()

    st.title("EyeV A&G Tool")
    st.caption("V7.13.3 prototype. Demo use only. Do not enter patient-identifiable information unless you have local approval.")

    with st.sidebar:
        st.header("Examples")
        selected_example = st.selectbox("Choose a test question", [""] + EXAMPLES)
        st.divider()
        st.subheader("Deployment note")
        st.write("For public demo links, set an app password and use synthetic or anonymised cases only.")
        st.divider()
        st.subheader("Logging")
        mode = logging_mode()
        if mode == "apps_script":
            st.success("Google Sheet logging configured")
        elif mode == "service_account":
            st.success("Google Sheet logging configured")
        else:
            st.info("Google Sheet logging not configured")

    default_text = selected_example or EXAMPLES[0]
    question = st.text_area("A&G request text", value=default_text, height=180)

    analyse = st.button("Analyse", type="primary")

    if analyse:
        cleaned = question.strip()
        if not cleaned:
            st.warning("Enter an A&G question first.")
            return

        result = engine.analyse(cleaned)
        log_status = "not_configured"
        try:
            log_status = log_to_google_sheet(cleaned, result)
        except Exception as exc:
            st.warning(f"Google Sheet logging failed: {exc}")

        st.subheader("Outcome recommendation")
        render_outcome(result["Outcome Recommendation"])

        st.subheader("Draft response")
        render_draft_response(result.get("Draft Response"))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detected features")
            render_features(result["Detected Features"])
        with col2:
            st.subheader("Safety")
            render_safety(result["Safety Ranking"])

        st.subheader("Presentation ranking")
        render_presentations(result["Presentation Ranking"])

        st.subheader("Missing information")
        render_missing_info(result["Missing Information"])

        with st.expander("Audit"):
            st.json(result["Audit"])

        if log_status == "logged":
            st.success("Logged to Google Sheet.")
        elif log_status == "not_configured":
            st.info("Google Sheet logging is not configured yet.")


if __name__ == "__main__":
    main()
