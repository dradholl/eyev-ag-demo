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
    "Needs clinician review",
    "Review reason",
    "Review status",
    "Reviewer category",
    "Clinician reviewer",
    "Clinician response",
    "Clinician outcome",
    "Graph update needed",
    "Proposed new feature",
    "Proposed new presentation",
    "Proposed missing info",
    "Proposed safety condition",
    "Validation case needed",
    "Outcome ID",
    "Outcome",
    "Outcome rationale",
    "Suggested response confidence",
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
    "Clinician final outcome",
    "Clinician final response",
    "Tool use status",
    "Override / edit reason",
    "Graph learning candidate",
    "Learning status",
    "Tool suggested outcome ID",
    "Tool suggested outcome",
    "Tool suggested draft response",
]

FEEDBACK_COLUMNS = [
    "Timestamp",
    "Question",
    "Tool suggested outcome ID",
    "Tool suggested outcome",
    "Tool suggested rationale",
    "Suggested response confidence",
    "Tool suggested draft response",
    "Clinician final outcome",
    "Clinician final response",
    "Tool use status",
    "Override / edit reason",
    "Graph learning candidate",
    "Learning status",
    "Detected feature IDs",
    "Detected features",
    "Top presentation ID",
    "Top presentation",
    "Safety IDs",
    "Safety conditions",
    "Missing information",
]

REVIEW_QUEUE_COLUMNS = [
    "Timestamp",
    "Question",
    "Needs clinician review",
    "Review reason",
    "Review status",
    "Reviewer category",
    "Outcome ID",
    "Outcome",
    "Top presentation ID",
    "Top presentation",
    "Detected feature IDs",
    "Detected features",
    "Missing information",
    "Draft suggested response",
    "Clinician reviewer",
    "Clinician response",
    "Clinician outcome",
    "Reviewer notes",
    "Graph update needed",
]

GRAPH_CANDIDATE_COLUMNS = [
    "Candidate timestamp",
    "Source question",
    "Candidate status",
    "Candidate reason",
    "Reviewer category",
    "Proposed change type",
    "Proposed new feature",
    "Proposed new presentation",
    "Proposed missing info",
    "Proposed safety condition",
    "Proposed validation case",
    "Clinician response",
    "Approval decision",
    "Approved by",
    "Implementation notes",
]

GRAPH_LEARNING_OPTIONS = [
    "Decide from my final response",
    "No, this was fine",
    "It missed wording in the request",
    "It suggested the wrong action",
    "It asked for unhelpful information",
    "This is a new topic or pathway",
    "There may be a safety issue",
]

OUTCOME_LABELS = {
    "OUT001": "Return to referrer with advice",
    "OUT002": "Return to referrer for more information",
    "OUT003": "Clinician to convert to referral",
}

OUTCOME_DISPLAY_LABELS = {
    "OUT001": "Send advice back",
    "OUT002": "Ask the referrer for more information",
    "OUT003": "Convert to referral",
}

LEARNING_SIGNAL_MAP = {
    "Decide from my final response": "Auto-detect from final response",
    "Work this out from my response": "Auto-detect from final response",
    "No, this was fine": "No",
    "No further improvement needed": "No",
    "The tool missed wording in the request": "Yes - missed phrase",
    "It missed important wording": "Yes - missed phrase",
    "It missed wording in the request": "Yes - missed phrase",
    "The tool suggested the wrong next step": "Yes - wrong outcome",
    "It suggested the wrong next step": "Yes - wrong outcome",
    "It suggested the wrong action": "Yes - wrong outcome",
    "The tool should have asked for different information": "Yes - missing information rule",
    "It asked for the wrong information": "Yes - missing information rule",
    "It asked for unhelpful information": "Yes - missing information rule",
    "This is a new topic or pathway": "Yes - new topic",
    "There may be a safety issue": "Yes - safety issue",
    "There may be a safety concern": "Yes - safety issue",
}


def a1_column_name(column_number):
    name = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        name = chr(65 + remainder) + name
    return name


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
        last_col = a1_column_name(len(SHEET_COLUMNS))
        worksheet.update(f"A1:{last_col}1", [SHEET_COLUMNS])

    return worksheet


def append_with_apps_script(sheets):
    url = st.secrets.get("GOOGLE_APPS_SCRIPT_URL", "")
    if not url:
        return False

    payload = {
        "token": st.secrets.get("GOOGLE_LOG_TOKEN", ""),
        "headers": sheets[0]["headers"] if sheets else [],
        "row": sheets[0]["row"] if sheets else [],
        "sheets": sheets,
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


def clinician_review_flags(result):
    if result.get("Review Override") == "glaucoma_adherence_advice":
        return "No", "Glaucoma adherence support advice", "Not required", "Existing pathway"

    outcome = result.get("Outcome Recommendation", {})
    rationale = outcome.get("Rationale", "")
    presentations = result.get("Presentation Ranking", [])
    features = result.get("Detected Features", [])
    missing_info = result.get("Missing Information", [])

    no_confident_presentation = not presentations
    no_features = not features
    unknown_query = "No recognised graph features" in rationale
    recognised_but_unmapped = "Recognised clinical features but no confident graph presentation" in rationale
    needs_more_info = outcome.get("Outcome ID") == "OUT002"

    if unknown_query or no_features:
        return "Yes", "Uncovered topic / no recognised graph features", "Unreviewed", "New topic"
    if recognised_but_unmapped or no_confident_presentation:
        return "Yes", "Recognised feature but no confident presentation", "Unreviewed", "Missed wording or pathway gap"
    if needs_more_info and missing_info:
        return "Optional", "More information requested by graph", "Awaiting referrer information", "Existing pathway"
    return "No", "Graph produced a covered pathway", "Not required", "Existing pathway"


def suggested_response_confidence(result):
    needs_review, review_reason, _status, _category = clinician_review_flags(result)
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_confidence = presentations[0].get("Confidence", 0) if presentations else 0

    if needs_review == "Yes":
        return "Low", "This topic is not yet covered confidently and should be checked by a clinician."

    if outcome.get("Outcome ID") == "OUT002":
        return "Medium", "The pathway is recognised, but more information is needed before final advice."

    if top_confidence >= 80:
        return "High", "The pathway is recognised with strong supporting features."

    if top_confidence >= 60:
        return "Medium", "The pathway is recognised, but the match is not strong."

    return "Low", "The match is uncertain and should be checked carefully."


def add_confidence_to_audit(result):
    confidence, explanation = suggested_response_confidence(result)
    result.setdefault("Audit", {})
    result["Audit"]["Suggested response confidence"] = confidence
    result["Audit"]["Suggested response confidence reason"] = explanation
    return result


def enforce_safe_review_outcome(result):
    needs_review, reason, _status, _category = clinician_review_flags(result)
    outcome = result.get("Outcome Recommendation", {})

    if needs_review == "Yes" and outcome.get("Outcome ID") == "OUT001":
        result["Outcome Recommendation"] = {
            "Outcome ID": "OUT002",
            "Outcome": OUTCOME_LABELS["OUT002"],
            "Rationale": f"Clinician review required: {reason}",
        }

    if needs_review == "Yes" and not result.get("Missing Information"):
        result["Missing Information"] = [{
            "Missing Information ID": "MI000",
            "Missing Information": "Main eye problem or question, which eye is affected, vision/VA, symptom duration, key symptoms, relevant examination findings and any photo/OCT/image if available",
        }]

    if needs_review == "Yes":
        result["Draft Response"] = {
            "Summary": "This topic needs clinician review before safe final advice can be given.",
            "Suggested response": (
                "Thanks for this. To help us advise safely, could you send a little more detail about the main eye problem, which eye is affected, "
                "vision/VA, how long it has been present, key symptoms, relevant examination findings and any photo/OCT/image if available?"
            ),
            "Safety net": "If there is pain, reduced vision, red eye, rapidly worsening symptoms, neurological symptoms or any other red flag, please refer urgently using the local ophthalmology pathway rather than waiting for advice.",
        }

    return result


def has_glaucoma_drop_adherence_context(question, result):
    text = question.lower()
    feature_ids = {feature.get("Feature ID", "") for feature in result.get("Detected Features", [])}
    return (
        "OF019" in feature_ids
        or (
            ("glaucoma drop" in text or "glaucoma drops" in text or "iop" in text or "iops" in text)
            and (
                "not been consistent" in text
                or "hasnt been consistent" in text
                or "hasn't been consistent" in text
                or "difficult" in text
                or "minims" in text
                or "adherence" in text
                or "compliance" in text
                or "instilling" in text
            )
        )
    )


def apply_glaucoma_drop_advice(question, result):
    if not has_glaucoma_drop_adherence_context(question, result):
        return result

    outcome = result.get("Outcome Recommendation", {})
    if outcome.get("Outcome ID") != "OUT003":
        result["Outcome Recommendation"] = {
            "Outcome ID": "OUT001",
            "Outcome": OUTCOME_LABELS["OUT001"],
            "Rationale": "Glaucoma drop adherence / IOP asymmetry support query.",
        }

    existing_missing = {
        item.get("Missing Information", "")
        for item in result.get("Missing Information", [])
    }
    additions = [
        ("MI010", "IOP values, timing, method and whether drops were used before measurement"),
        ("MI015", "Current glaucoma drop name, prescribed frequency, adherence and instillation difficulty"),
        ("MI012", "Optic disc/OCT/visual field status, reliability and progression"),
        ("MI061", "Current HES/glaucoma follow-up status and whether the patient has new symptoms"),
    ]
    result.setdefault("Missing Information", [])
    for missing_id, text in additions:
        if text not in existing_missing:
            result["Missing Information"].append({
                "Missing Information ID": missing_id,
                "Missing Information": text,
            })

        result["Draft Response"] = {
            "Summary": "This appears to be a glaucoma drop adherence / IOP review query.",
            "Suggested response": (
                "Thanks for this. From the information provided, this sounds primarily like difficulty with glaucoma drop instillation/adherence rather than an acute glaucoma presentation. "
                "Reasonable support would be to check drop technique, simplify the practical routine where possible, involve a carer if appropriate, and consider a compliance aid/drop dispenser."
            ),
            "Safety net": (
                "If IOP is very high, there is eye pain/redness, corneal haze, halos, nausea/vomiting, sudden vision loss, or marked progression/new symptoms, "
                "please refer urgently using the local glaucoma/ophthalmology pathway."
            ),
        }
    result["Review Override"] = "glaucoma_adherence_advice"
    return result


def normalise_outcome_label(result):
    outcome = result.get("Outcome Recommendation", {})
    outcome_id = outcome.get("Outcome ID", "")
    if outcome_id in OUTCOME_LABELS:
        outcome["Outcome"] = OUTCOME_LABELS[outcome_id]
        result["Outcome Recommendation"] = outcome
    return result


def ensure_draft_response(question, result):
    if result.get("Draft Response"):
        return result

    outcome = result.get("Outcome Recommendation", {})
    missing_info = result.get("Missing Information", [])
    presentations = result.get("Presentation Ranking", [])
    top = presentations[0] if presentations else {}

    if outcome.get("Outcome ID") == "OUT002":
        info = ", ".join(item.get("Missing Information", "") for item in missing_info if item.get("Missing Information"))
        if not info:
            info = "clinical question, laterality, VA, symptom duration, relevant positive/negative symptoms, examination findings and images/OCT/photos where available"
        result["Draft Response"] = {
            "Summary": "More information is needed before safe advice can be given.",
            "Suggested response": f"Thanks for this. I can advise more safely with a little more detail. Could you send {info}?",
            "Safety net": "If there are urgent symptoms or red flags, please refer urgently using the local ophthalmology pathway rather than waiting for advice.",
        }
    elif outcome.get("Outcome ID") == "OUT003":
        result["Draft Response"] = {
            "Summary": "The information provided suggests ophthalmology assessment is needed.",
            "Suggested response": (
                "Thanks for this. From the details provided, this needs ophthalmology assessment. "
                "Please refer using the appropriate local pathway and include symptom onset, VA, laterality, key positive and negative symptoms, relevant examination findings and any images/OCT/photos available."
            ),
            "Safety net": "If symptoms are acute, severe, rapidly worsening, or associated with pain, red eye, neurological symptoms or sudden reduced vision, please refer urgently rather than via a routine pathway.",
        }
    else:
        presentation_text = top.get("Presentation", "the supplied information")
        result["Draft Response"] = {
            "Summary": f"This looks suitable for advice back to the referrer based on {presentation_text}.",
            "Suggested response": (
                "Thanks for this. From the information provided, this sounds suitable for advice back to the referrer, provided there are no red-flag symptoms or concerning examination findings. "
                "If key clinical details are missing, please request those details before issuing final advice."
            ),
            "Safety net": "Please advise the patient to seek urgent reassessment if symptoms worsen, vision drops, pain/redness develops, or any other red flags appear.",
        }

    return result


def soften_draft_response_language(result):
    draft = result.get("Draft Response")
    if not draft:
        return result

    response = str(draft.get("Suggested response", "") or "").strip()
    safety_net = str(draft.get("Safety net", "") or "").strip()
    query = normalise_text(result.get("Query", ""))
    presentations = result.get("Presentation Ranking", [])
    top_presentation_id = presentations[0].get("Presentation ID", "") if presentations else ""

    if (
        top_presentation_id == "PR051"
        or "optic disc drusen" in query
        or "pseudopapilloedema" in query
        or "pseudopapilledema" in query
    ):
        draft["Suggested response"] = (
            "Thanks for this. If the optic disc drusen/pseudopapilloedema is already known, stable and there are no new symptoms or new examination concerns, "
            "a new ophthalmology referral does not appear to be needed on the information provided. Please advise the patient to seek review if they develop new visual symptoms or headaches, "
            "or if the disc appearance changes."
        )
        draft["Safety net"] = (
            "If there are new headaches, transient visual obscurations, diplopia, vomiting, reduced vision, field loss, or concern for true disc swelling, "
            "please refer urgently using the local ophthalmology pathway."
        )
        result["Draft Response"] = draft
        return result

    if top_presentation_id == "PR010" or (
        ("raised iop" in query or "iop" in query or "iops" in query)
        and ("disc" in query or "glaucoma" in query or "field" in query)
    ):
        draft["Suggested response"] = (
            "Thanks for this. The raised IOP and suspicious disc findings need glaucoma assessment, so please refer via the local glaucoma pathway. "
            "It would be helpful to include IOP values and method, disc/OCT RNFL images, visual-field printouts/reliability, previous comparison and whether the patient is already under HES/glaucoma follow-up."
        )
        draft["Safety net"] = (
            "If there are acute angle-closure symptoms, sudden vision loss, painful red eye or rapidly progressive field/optic-nerve change, "
            "please refer urgently using the local pathway."
        )
        result["Draft Response"] = draft
        return result

    if response.startswith("Return advice only"):
        draft["Suggested response"] = (
            "Thanks for this. From the information provided, this sounds suitable for advice back to the referrer, provided there are no red-flag symptoms or concerning examination findings. "
            "Please safety-net the patient to seek urgent reassessment if symptoms worsen, vision drops, pain/redness develops, or any other red flags appear."
        )

    if response.startswith("Convert or escalate"):
        draft["Suggested response"] = (
            "Thanks for this. From the details provided, this should be referred for ophthalmology assessment using the appropriate local pathway. "
            "Please include the symptom history, VA, laterality, relevant positive and negative findings, and any images/OCT/photos available."
        )

    if response.startswith("Please provide:"):
        details = response.replace("Please provide:", "", 1).strip().rstrip(".")
        if details:
            draft["Suggested response"] = (
                "Thanks for this. I can advise more safely with a little more detail. "
                f"Could you send {details}"
            )
            if not draft["Suggested response"].endswith((".", "?")):
                draft["Suggested response"] += "?"

    elif response.startswith("Please provide "):
        details = response.replace("Please provide ", "", 1).strip().rstrip(".")
        if details:
            draft["Suggested response"] = (
                "Thanks for this. I can advise more safely with a little more detail. "
                f"Could you send {details}"
            )
            if not draft["Suggested response"].endswith((".", "?")):
                draft["Suggested response"] += "?"

    if "use the relevant urgent pathway" in safety_net:
        draft["Safety net"] = safety_net.replace(
            "use the relevant urgent pathway",
            "please refer urgently using the local ophthalmology pathway",
        )
    if "please use the relevant urgent pathway" in safety_net:
        draft["Safety net"] = safety_net.replace(
            "please use the relevant urgent pathway",
            "please refer urgently using the local ophthalmology pathway",
        )

    result["Draft Response"] = draft
    return result


def prepare_result_for_display(question, result):
    result = apply_glaucoma_drop_advice(question, result)
    result = enforce_safe_review_outcome(result)
    result = ensure_draft_response(question, result)
    result = soften_draft_response_language(result)
    result = add_confidence_to_audit(result)
    result = normalise_outcome_label(result)
    return result


def empty_feedback():
    return {
        "clinician_final_outcome": "",
        "clinician_final_response": "",
        "tool_use_status": "",
        "override_reason": "",
        "graph_learning_candidate": "",
        "learning_status": "",
        "reasoning_not_satisfactory": False,
    }


def outcome_id_from_label(label):
    label = str(label or "").strip()
    if label.startswith("OUT"):
        return label.split(":", 1)[0].strip()
    for outcome_id, display_label in OUTCOME_DISPLAY_LABELS.items():
        if label == display_label:
            return outcome_id
    for outcome_id, full_label in OUTCOME_LABELS.items():
        if label == full_label:
            return outcome_id
    return label


def outcome_log_label(display_label):
    outcome_id = outcome_id_from_label(display_label)
    if outcome_id in OUTCOME_LABELS:
        return f"{outcome_id}: {OUTCOME_LABELS[outcome_id]}"
    return str(display_label or "")


def internal_learning_signal(label):
    return LEARNING_SIGNAL_MAP.get(str(label or "").strip(), str(label or "").strip())


def normalise_text(value):
    return " ".join(str(value or "").lower().split())


def infer_tool_use_status(result, feedback):
    if feedback.get("reasoning_not_satisfactory"):
        return "Rejected - reasoning concern"

    outcome = result.get("Outcome Recommendation", {})
    draft = result.get("Draft Response", {})
    tool_outcome_id = outcome.get("Outcome ID", "")
    clinician_outcome_id = outcome_id_from_label(feedback.get("clinician_final_outcome", ""))
    tool_response = normalise_text(draft.get("Suggested response", ""))
    clinician_response = normalise_text(feedback.get("clinician_final_response", ""))

    if clinician_outcome_id and clinician_outcome_id != tool_outcome_id:
        return "Overridden"
    if clinician_response != tool_response:
        return "Edited"
    return "Accepted"


def clinician_response_suggests_referral(text):
    text = normalise_text(text)
    referral_terms = [
        "convert",
        "converted",
        "refer",
        "referral",
        "book",
        "clinic",
        "urgent",
        "same day",
        "eye casualty",
        "ophthalmology",
        "oculoplastics",
        "medical retina",
        "glaucoma",
    ]
    return any(term in text for term in referral_terms)


def clinician_response_requests_information(text):
    text = normalise_text(text)
    info_terms = [
        "please provide",
        "provide",
        "attach",
        "send",
        "photo",
        "photograph",
        "image",
        "oct",
        "visual acuity",
        " va ",
        "duration",
        "onset",
        "laterality",
        "which eye",
        "size",
        "growth",
        "change",
        "bleeding",
        "ulceration",
        "lash loss",
        "redness",
        "pain",
        "iop",
        "fields",
    ]
    padded = f" {text} "
    return any(term in padded for term in info_terms)


def infer_graph_learning_candidate(question, result, feedback):
    requested = internal_learning_signal(feedback.get("graph_learning_candidate", ""))
    if requested and requested != "Auto-detect from final response":
        return requested, feedback.get("override_reason", "")

    outcome = result.get("Outcome Recommendation", {})
    tool_outcome_id = outcome.get("Outcome ID", "")
    clinician_outcome_id = outcome_id_from_label(feedback.get("clinician_final_outcome", ""))
    response = feedback.get("clinician_final_response", "")
    override_reason = feedback.get("override_reason", "")
    needs_review, review_reason, _status, category = clinician_review_flags(result)
    presentations = result.get("Presentation Ranking", [])
    features = result.get("Detected Features", [])

    if feedback.get("reasoning_not_satisfactory"):
        return "Yes - missed phrase", "Clinician marked the tool reasoning as not clinically satisfactory."

    if clinician_outcome_id and clinician_outcome_id != tool_outcome_id:
        if clinician_outcome_id == "OUT003" or clinician_response_suggests_referral(response):
            return "Yes - wrong outcome", "Clinician final outcome was more escalatory than the tool."
        return "Yes - wrong outcome", "Clinician final outcome differed from the tool."

    if needs_review == "Yes":
        if not features or category == "New topic":
            return "Yes - new topic", review_reason
        if not presentations:
            return "Yes - missed phrase", review_reason
        return "Yes - missed phrase", review_reason

    if clinician_response_requests_information(response) and outcome.get("Outcome ID") == "OUT001":
        return "Yes - missing information rule", "Clinician response asked for extra details when the tool suggested advice."

    if clinician_response_suggests_referral(response) and outcome.get("Outcome ID") != "OUT003":
        return "Yes - safety issue", "Clinician response suggested referral/escalation when the tool did not."

    if override_reason:
        return "Yes - missed phrase", override_reason

    return "No", ""


def extract_likely_topic_phrase(question, result):
    text = normalise_text(question)
    feature_names = [feature.get("Feature", "") for feature in result.get("Detected Features", [])]
    topic_terms = [
        ("lid lump / suspicious eyelid lesion", ["lid", "lump", "eyelid", "lesion", "waterline", "lash"]),
        ("macular OCT / retinal lesion", ["macula", "macular", "oct", "retina", "retinal", "fluid", "hole"]),
        ("glaucoma / IOP / optic nerve", ["glaucoma", "iop", "pressure", "optic nerve", "disc", "field"]),
        ("cornea / keratoconus", ["cornea", "corneal", "keratoconus", "contact lens"]),
        ("watering eye / lacrimal", ["watering", "watery", "epiphora", "lacrimal", "tear duct"]),
    ]
    for label, terms in topic_terms:
        if any(term in text for term in terms):
            return label
    if feature_names:
        return "; ".join(feature_names[:3])
    words = [word for word in text.split() if len(word) > 4]
    return " ".join(words[:8]) or "Unclear clinical topic"


def extract_missing_information_hint(response):
    text = normalise_text(response)
    hints = []
    checks = [
        ("clinical photograph / image", ["photo", "photograph", "image", "attach"]),
        ("duration / onset / change", ["duration", "onset", "change", "growth", "worsening"]),
        ("VA and laterality", ["visual acuity", " va ", "laterality", "which eye"]),
        ("red-flag symptoms", ["pain", "redness", "photophobia", "bleeding", "ulceration", "lash loss"]),
        ("OCT / scan findings", ["oct", "scan"]),
        ("IOP / disc / fields", ["iop", "pressure", "disc", "field", "rnfl"]),
    ]
    padded = f" {text} "
    for label, terms in checks:
        if any(term in padded for term in terms):
            hints.append(label)
    return "; ".join(hints)


def suggested_validation_case(question, feedback):
    outcome_id = outcome_id_from_label(feedback.get("clinician_final_outcome", ""))
    response = feedback.get("clinician_final_response", "")
    if not question:
        return ""
    return f"Query: {question} | Expected outcome: {outcome_id or 'review'} | Clinician response: {response[:180]}"


def graph_learning_payload(question, result, feedback):
    learning_candidate, inferred_reason = infer_graph_learning_candidate(question, result, feedback)
    response = feedback.get("clinician_final_response", "")
    outcome_id = outcome_id_from_label(feedback.get("clinician_final_outcome", ""))
    topic_phrase = extract_likely_topic_phrase(question, result)
    missing_hint = extract_missing_information_hint(response)

    if learning_candidate == "No":
        return {
            "learning_candidate": "No",
            "learning_status": "No learning action",
            "candidate_reason": inferred_reason,
            "proposed_change_type": "",
            "proposed_new_feature": "",
            "proposed_new_presentation": "",
            "proposed_missing_info": "",
            "proposed_safety_condition": "",
            "proposed_validation_case": "",
        }

    if learning_candidate == "Yes - new topic":
        proposed_change_type = "New feature/presentation"
    elif learning_candidate == "Yes - missed phrase":
        proposed_change_type = "Synonym/pathway review"
    elif learning_candidate == "Yes - missing information rule":
        proposed_change_type = "Missing-information rule"
    elif learning_candidate == "Yes - safety issue":
        proposed_change_type = "Safety/outcome review"
    else:
        proposed_change_type = "Outcome calibration"

    proposed_new_presentation = ""
    if outcome_id:
        proposed_new_presentation = f"{topic_phrase} -> {outcome_id}"

    return {
        "learning_candidate": learning_candidate,
        "learning_status": "Pending review",
        "candidate_reason": inferred_reason,
        "proposed_change_type": proposed_change_type,
        "proposed_new_feature": topic_phrase,
        "proposed_new_presentation": proposed_new_presentation,
        "proposed_missing_info": missing_hint,
        "proposed_safety_condition": "Review if clinician response indicates referral/escalation or red flags",
        "proposed_validation_case": suggested_validation_case(question, feedback),
    }


def result_to_sheet_row(question, result, feedback=None):
    feedback = feedback or empty_feedback()
    learning = graph_learning_payload(question, result, feedback)
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_presentation = presentations[0] if presentations else {}
    draft = result.get("Draft Response", {})
    needs_review, review_reason, review_status, reviewer_category = clinician_review_flags(result)
    confidence, _confidence_reason = suggested_response_confidence(result)

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        needs_review,
        review_reason,
        review_status,
        reviewer_category,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        outcome.get("Rationale", ""),
        confidence,
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
        feedback.get("clinician_final_outcome", ""),
        feedback.get("clinician_final_response", ""),
        feedback.get("tool_use_status", ""),
        feedback.get("override_reason", ""),
        learning["learning_candidate"],
        learning["learning_status"],
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        draft.get("Suggested response", ""),
    ]


def result_to_review_queue_row(question, result, feedback=None):
    feedback = feedback or empty_feedback()
    learning = graph_learning_payload(question, result, feedback)
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_presentation = presentations[0] if presentations else {}
    draft = result.get("Draft Response", {})
    needs_review, review_reason, review_status, reviewer_category = clinician_review_flags(result)

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        needs_review,
        review_reason,
        review_status,
        reviewer_category,
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        top_presentation.get("Presentation ID", ""),
        top_presentation.get("Presentation", ""),
        join_values(result.get("Detected Features", []), "Feature ID"),
        join_values(result.get("Detected Features", []), "Feature"),
        join_values(result.get("Missing Information", []), "Missing Information"),
        draft.get("Suggested response", ""),
        "",
        feedback.get("clinician_final_response", ""),
        feedback.get("clinician_final_outcome", ""),
        "",
        learning["proposed_change_type"] or "",
    ]


def result_to_graph_candidate_row(question, result, feedback=None):
    feedback = feedback or empty_feedback()
    learning = graph_learning_payload(question, result, feedback)
    draft = result.get("Draft Response", {})
    needs_review, review_reason, _review_status, reviewer_category = clinician_review_flags(result)
    proposed_change_type = learning["proposed_change_type"] or ("New pathway" if reviewer_category == "New topic" else "Synonym/pathway review")
    candidate_reason = learning["candidate_reason"] or review_reason

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        "Pending clinician review",
        candidate_reason,
        reviewer_category,
        proposed_change_type,
        learning["proposed_new_feature"],
        learning["proposed_new_presentation"],
        learning["proposed_missing_info"],
        learning["proposed_safety_condition"],
        learning["proposed_validation_case"],
        feedback.get("clinician_final_response", "") or draft.get("Suggested response", ""),
        "",
        "",
        "",
    ]


def result_to_feedback_row(question, result, feedback):
    learning = graph_learning_payload(question, result, feedback)
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_presentation = presentations[0] if presentations else {}
    draft = result.get("Draft Response", {})
    confidence, _confidence_reason = suggested_response_confidence(result)

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        outcome.get("Rationale", ""),
        confidence,
        draft.get("Suggested response", ""),
        feedback.get("clinician_final_outcome", ""),
        feedback.get("clinician_final_response", ""),
        feedback.get("tool_use_status", ""),
        feedback.get("override_reason", ""),
        learning["learning_candidate"],
        learning["learning_status"],
        join_values(result.get("Detected Features", []), "Feature ID"),
        join_values(result.get("Detected Features", []), "Feature"),
        top_presentation.get("Presentation ID", ""),
        top_presentation.get("Presentation", ""),
        join_values(result.get("Safety Ranking", []), "Safety Condition ID"),
        join_values(result.get("Safety Ranking", []), "Safety Condition"),
        join_values(result.get("Missing Information", []), "Missing Information"),
    ]


def build_logging_payloads(question, result, feedback=None):
    feedback = feedback or empty_feedback()
    learning = graph_learning_payload(question, result, feedback)
    row = result_to_sheet_row(question, result, feedback)
    sheets = [
        {
            "name": "A&G Log",
            "headers": SHEET_COLUMNS,
            "row": row,
        },
        {
            "name": "Clinician Review Queue",
            "headers": REVIEW_QUEUE_COLUMNS,
            "row": [],
        },
        {
            "name": "Graph Update Candidates",
            "headers": GRAPH_CANDIDATE_COLUMNS,
            "row": [],
        },
        {
            "name": "Clinician Feedback",
            "headers": FEEDBACK_COLUMNS,
            "row": result_to_feedback_row(question, result, feedback),
        },
    ]

    needs_review, _reason, _status, _category = clinician_review_flags(result)
    learning_candidate = learning["learning_candidate"]
    if needs_review == "Yes" or learning_candidate.startswith("Yes"):
        sheets[1]["row"] = result_to_review_queue_row(question, result, feedback)
        sheets[2]["row"] = result_to_graph_candidate_row(question, result, feedback)

    return sheets


def log_to_google_sheet(question, result, feedback=None):
    sheets = build_logging_payloads(question, result, feedback)

    if st.secrets.get("GOOGLE_APPS_SCRIPT_URL", ""):
        append_with_apps_script(sheets)
        return "logged"

    worksheet = load_google_sheet()
    if worksheet is None:
        return "not_configured"

    worksheet.append_row(
        sheets[0]["row"],
        value_input_option="USER_ENTERED",
    )
    return "logged"


def diagnostic_log_row():
    row = [""] * len(SHEET_COLUMNS)
    values = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Question": "DIAGNOSTIC TEST - Streamlit to Google Sheet logging",
        "Needs clinician review": "No",
        "Review reason": "Diagnostic test row",
        "Review status": "Not required",
        "Reviewer category": "System test",
        "Outcome ID": "TEST",
        "Outcome": "Diagnostic logging test",
        "Outcome rationale": "Manual sidebar test",
        "Suggested response confidence": "High",
        "Draft summary": "Diagnostic test",
        "Draft suggested response": "If this row appears, Streamlit can write to the Google Sheet.",
        "Tool suggested outcome ID": "TEST",
        "Tool suggested outcome": "Diagnostic logging test",
        "Tool suggested draft response": "If this row appears, Streamlit can write to the Google Sheet.",
    }
    for column, value in values.items():
        row[SHEET_COLUMNS.index(column)] = value
    return row


def run_logging_diagnostic():
    if not st.secrets.get("GOOGLE_APPS_SCRIPT_URL", ""):
        return "Google Apps Script URL is not configured in Streamlit secrets."

    append_with_apps_script([{
        "name": "A&G Log",
        "headers": SHEET_COLUMNS,
        "row": diagnostic_log_row(),
    }])
    return "Diagnostic row sent to Google Sheet."


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


def render_clinician_review_notice(result):
    needs_review, reason, status, category = clinician_review_flags(result)
    if needs_review == "Yes":
        st.warning(
            "Clinician review required before final advice. "
            f"Reason: {reason}. Category: {category}."
        )
    elif needs_review == "Optional":
        st.info(
            "This case is on an existing pathway but needs more information. "
            "Clinician review may be useful if the referrer cannot provide the requested details."
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


def render_confidence_badge(confidence):
    styles = {
        "High": ("✓", "#e6f4ea", "#137333"),
        "Medium": ("!", "#fff4e5", "#a15c00"),
        "Low": ("!", "#fdecea", "#b3261e"),
    }
    icon, background, colour = styles.get(confidence, ("?", "#eef2f7", "#405261"))
    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 0.55rem; padding: 0.45rem 0.7rem; border-radius: 999px; background: {background}; color: {colour}; font-weight: 700; margin: 0.25rem 0 0.75rem 0;">
          <span style="display: inline-flex; align-items: center; justify-content: center; width: 1.35rem; height: 1.35rem; border-radius: 999px; border: 2px solid {colour}; font-size: 0.95rem; line-height: 1;">{icon}</span>
          <span>{confidence} confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clinician_feedback_form(question, result):
    outcome = result.get("Outcome Recommendation", {})
    draft = result.get("Draft Response", {})
    needs_review, review_reason, _status, _category = clinician_review_flags(result)
    tool_outcome_id = outcome.get("Outcome ID", "")
    outcome_options = [
        OUTCOME_DISPLAY_LABELS[outcome_id]
        for outcome_id in ("OUT001", "OUT002", "OUT003")
    ]
    default_outcome = next(
        (idx for idx, option in enumerate(outcome_options) if option == OUTCOME_DISPLAY_LABELS.get(tool_outcome_id, "")),
        0,
    )
    confidence, _confidence_reason = suggested_response_confidence(result)

    st.subheader("Clinician response")
    st.caption("Check the suggested wording, amend it if needed, then save the response.")
    render_confidence_badge(confidence)
    with st.container(border=True):
        clinician_final_outcome = st.radio(
            "What should happen next?",
            outcome_options,
            index=default_outcome,
            horizontal=True,
        )
        clinician_final_response = st.text_area(
            "Response to send",
            value=draft.get("Suggested response", ""),
            height=140,
            placeholder=(
                "A few words is enough, e.g. 'Refer to oculoplastics' or "
                "'Ask for lid photo, duration, change, bleeding/lash loss'."
            ),
        )

        suggestion_helpful = st.radio(
            "Was the suggested response helpful?",
            ["👍 Helpful", "👎 Needs improvement"],
            horizontal=True,
        )
        graph_learning_candidate = "No, this was fine"
        override_reason = ""
        reasoning_not_satisfactory = False

        if suggestion_helpful == "👎 Needs improvement":
            reasoning_not_satisfactory = True
            graph_learning_candidate = st.selectbox(
                "What would improve it?",
                GRAPH_LEARNING_OPTIONS,
                index=0,
            )
            override_reason = st.text_area(
                "What was missing or unhelpful?",
                value="",
                height=120,
                placeholder="A few words is enough. For example: missed lid lump wording, should have suggested referral, or should ask for a photo.",
            )

        submitted = st.button("Save clinician response", type="primary")

    if not submitted:
        return

    feedback = {
        "clinician_final_outcome": outcome_log_label(clinician_final_outcome),
        "clinician_final_response": clinician_final_response.strip(),
        "tool_use_status": "",
        "override_reason": override_reason.strip(),
        "graph_learning_candidate": internal_learning_signal(graph_learning_candidate),
        "learning_status": "",
        "reasoning_not_satisfactory": reasoning_not_satisfactory,
    }
    feedback["tool_use_status"] = infer_tool_use_status(result, feedback)
    learning = graph_learning_payload(question, result, feedback)
    feedback["graph_learning_candidate"] = learning["learning_candidate"]
    feedback["learning_status"] = learning["learning_status"]

    try:
        log_status = log_to_google_sheet(question, result, feedback)
    except Exception as exc:
        st.warning(f"Google Sheet logging failed: {exc}")
        return

    if log_status == "logged":
        st.success("Response saved.")
        if learning["learning_candidate"].startswith("Yes"):
            st.info(
                "This case has been added to the review list for future improvement."
            )
    elif log_status == "not_configured":
        st.info("Google Sheet logging is not configured yet.")


def main():
    if not check_password():
        return

    engine = load_engine()

    st.title("EyeV A&G Tool")
    st.caption("Clinician support prototype. Demo use only. Do not enter patient-identifiable information unless you have local approval.")

    with st.sidebar:
        st.header("Example cases")
        selected_example = st.selectbox("Try an example", [""] + EXAMPLES)
        st.divider()
        with st.expander("Admin"):
            st.write("For public demo links, set an app password and use synthetic or anonymised cases only.")
            mode = logging_mode()
            if mode in ("apps_script", "service_account"):
                st.success("Google Sheet logging configured")
            else:
                st.info("Google Sheet logging not configured")

            if st.button("Test Google Sheet logging"):
                try:
                    st.success(run_logging_diagnostic())
                except Exception as exc:
                    st.error(f"Logging test failed: {exc}")

    default_text = selected_example or EXAMPLES[0]
    question = st.text_area("A&G request text", value=default_text, height=180)

    analyse = st.button("Create suggested response", type="primary")

    if analyse:
        cleaned = question.strip()
        if not cleaned:
            st.warning("Enter an A&G question first.")
            return

        result = engine.analyse(cleaned)
        result = prepare_result_for_display(cleaned, result)
        st.session_state["last_question"] = cleaned
        st.session_state["last_result"] = result

    if st.session_state.get("last_result"):
        cleaned = st.session_state["last_question"]
        result = st.session_state["last_result"]
        render_clinician_feedback_form(cleaned, result)


if __name__ == "__main__":
    main()
