"""Streamlit demo app for the EyeV A&G reasoning engine."""

import base64
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from EyeV_OKG_V7_engine import OKGEngine


APP_DIR = Path(__file__).resolve().parent
GRAPH_FILE = APP_DIR / "EyeV_Ophthalmic_Knowledge_Graph_v2.xlsx"
OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
IMAGE_REVIEW_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
IMAGE_REVIEW_PROMPT_VERSION = "ag-image-validation-v1"
IMAGE_REVIEW_MAX_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_REVIEW_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/pdf",
}

IMAGE_REVIEW_LABELS = {
    "image_accessible": ["Yes", "No", "Unclear"],
    "image_type": [
        "OCT macula",
        "OCT RNFL/disc",
        "Fundus photo",
        "Visual field",
        "External eye / lids",
        "Anterior segment photo",
        "Report / text document",
        "Other",
        "Unclear",
    ],
    "image_quality": ["Good", "Adequate", "Poor", "Unclear"],
    "finding_visible": ["Yes", "Partly", "No", "Unclear"],
    "finding_category": [
        "Macula/OCT",
        "Optic disc/RNFL",
        "Glaucoma/fields",
        "External eye/lids",
        "Anterior segment",
        "Retina lesion",
        "Normal/no obvious abnormality",
        "Other",
        "Unclear",
    ],
    "supports_request": ["Yes", "Partly", "No", "Unclear"],
    "confidence": ["High", "Medium", "Low"],
}

IMAGE_REVIEW_SYSTEM_PROMPT = """You are supporting an ophthalmology advice-and-guidance image workflow.

You are not making an autonomous diagnosis or management decision. Your role is to describe what is visible in the attached image/file and judge whether the image appears relevant to the clinical question asked by the referrer.

Use only the allowed labels. Be cautious where image quality, file access, or clinical context is limited. If the attachment is not visible to you, say image_accessible = No or Unclear and keep other labels conservative.

The support label means whether the image is relevant to, and helps assess, the A&G request. It does not mean the historical clinician response was correct and it must not be treated as an automated final decision.

Return JSON only."""

IMAGE_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_accessible": {"type": "string", "enum": IMAGE_REVIEW_LABELS["image_accessible"]},
        "image_type": {"type": "string", "enum": IMAGE_REVIEW_LABELS["image_type"]},
        "image_quality": {"type": "string", "enum": IMAGE_REVIEW_LABELS["image_quality"]},
        "finding_visible": {"type": "string", "enum": IMAGE_REVIEW_LABELS["finding_visible"]},
        "finding_category": {"type": "string", "enum": IMAGE_REVIEW_LABELS["finding_category"]},
        "brief_image_finding": {
            "type": "string",
            "description": "Short plain-English description of visible finding or limitation.",
            "maxLength": 500,
        },
        "supports_ag_request": {"type": "string", "enum": IMAGE_REVIEW_LABELS["supports_request"]},
        "confidence": {"type": "string", "enum": IMAGE_REVIEW_LABELS["confidence"]},
        "limitations": {
            "type": "string",
            "description": "Short note on visibility, uncertainty, or access limitations.",
            "maxLength": 500,
        },
    },
    "required": [
        "image_accessible",
        "image_type",
        "image_quality",
        "finding_visible",
        "finding_category",
        "brief_image_finding",
        "supports_ag_request",
        "confidence",
        "limitations",
    ],
}


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
    "Suggested response confidence reason",
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
    "Suggested response helpful",
    "Clinical reasoning concern",
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
    "Suggested response confidence reason",
    "Tool suggested draft response",
    "Clinician final outcome",
    "Clinician final response",
    "Suggested response helpful",
    "Clinical reasoning concern",
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
    "Suggested response confidence",
    "Suggested response confidence reason",
    "Top presentation ID",
    "Top presentation",
    "Detected feature IDs",
    "Detected features",
    "Missing information",
    "Draft suggested response",
    "Clinician reviewer",
    "Clinician response",
    "Clinician outcome",
    "Suggested response helpful",
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

IMAGE_REVIEW_COLUMNS = [
    "Image attached",
    "Image filename",
    "Image content type",
    "Image size bytes",
    "GPT image model",
    "GPT image prompt version",
    "GPT image accessible",
    "GPT image type",
    "GPT image quality",
    "GPT finding visible",
    "GPT finding category",
    "GPT brief image finding",
    "GPT image relevance to A&G request",
    "GPT image confidence",
    "GPT image uncertainty flag",
    "GPT image limitations",
    "GPT image error",
    "Clinician image summary assessment",
    "Clinician image notes",
]

for column in IMAGE_REVIEW_COLUMNS:
    if column not in SHEET_COLUMNS:
        SHEET_COLUMNS.append(column)
    if column not in FEEDBACK_COLUMNS:
        FEEDBACK_COLUMNS.append(column)
    if column not in REVIEW_QUEUE_COLUMNS:
        REVIEW_QUEUE_COLUMNS.append(column)

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
    "OUT003": "Clinician converts to referral",
}

PLAIN_MISSING_INFO_LABELS = {
    "Visual acuity": "current vision/VA",
    "Duration of symptoms": "when the symptoms started and whether they are changing",
    "Field defect status": "whether there is any curtain, shadow or missing area of vision",
    "Dilated fundus view": "whether a dilated retinal examination was performed and what was seen",
    "OCT image / scan attached": "the OCT image or scan report",
    "Laterality": "which eye is affected",
    "Distortion / Amsler status": "whether there is distortion, wavy vision or an abnormal Amsler result",
    "Previous AMD / macular history": "any previous AMD, macular history, injections or comparison OCT",
    "Fundus findings / macular appearance": "fundus findings, including any haemorrhage, exudate or macular abnormality",
    "IOP values and method": "IOP readings, which eye, how they were measured and whether they were repeated",
    "Optic disc assessment / images": "disc appearance, C/D ratio and any OCT RNFL or disc photos",
    "Visual field test reliability / progression": "visual-field results, reliability and whether any defect is repeatable or progressing",
    "Angle closure symptoms": "whether there is eye pain, halos, nausea/vomiting, headache, red eye or pupil symptoms",
    "Glaucoma family history / risk factors": "family history and other glaucoma risk factors",
    "Current glaucoma drops / adherence": "current glaucoma drops, prescribed frequency, adherence and any difficulty instilling them",
    "Contact lens wear status": "whether the patient wears contact lenses and whether they have stopped wearing them",
    "Corneal staining / infiltrate / opacity": "fluorescein staining, epithelial defect, infiltrate or corneal opacity findings",
    "Photophobia and pain severity": "whether there is light sensitivity and how severe the pain is",
    "Anterior chamber reaction": "whether there are cells/flare or anterior chamber inflammation",
    "Discharge and infective symptoms": "whether there is discharge, sticky eye, watering or infective symptoms",
    "Functional impact / driving and glare symptoms": "impact on daily activities, reading, driving, glare or work",
    "Cataract laterality and severity": "which eye is affected and the cataract grade/severity",
    "Previous cataract surgery date / lens status": "date of cataract surgery, which eye was operated on and lens implant/pseudophakia details",
    "Post-operative pain/redness/vision status": "post-operative vision and whether there is pain, redness, discharge or light sensitivity",
    "Ocular comorbidity / fundus view": "macular, glaucoma or other eye comorbidity and fundus view findings",
    "Lid lesion photograph": "a clinical photograph of the eyelid lesion or lid position",
    "Lid lesion duration and growth": "how long the lid lesion has been present and whether it is growing, changing or recurrent",
    "Lid lesion red flags": "whether there is ulceration, bleeding, lash loss, pigmentation, irregular margin or pain",
    "Orbital/neuro symptoms": "whether there is proptosis, double vision, pain on eye movement, restricted eye movement, reduced vision or systemic symptoms",
    "Effect on vision / ocular surface": "whether the lid problem affects vision, cornea, tearing, exposure or the ocular surface",
    "Diabetic retinopathy grade / screening result": "diabetic retinopathy/maculopathy grade or screening result",
    "Macular OCT / DMO status": "OCT findings and whether there is diabetic macular oedema or macular thickening",
    "Diabetes control / systemic risk context": "HbA1c, blood pressure, duration of diabetes and relevant systemic risk factors",
    "Previous diabetic retina treatment": "previous PRP, focal laser, injections or retina clinic treatment",
    "Diplopia onset and pattern": "onset, duration and whether the double vision is monocular/binocular or variable",
    "Ocular motility findings": "eye movement findings, cover test/deviation and affected gaze direction",
    "Neurological symptoms / pupil / ptosis": "whether there is headache, neurological symptoms, pupil abnormality, ptosis or pain",
    "Child age and visual acuity / amblyopia status": "child's age, VA/fixation, amblyopia history and previous orthoptic care",
    "Current visual acuity and functional impact": "current VA and effect on daily activities",
    "Diagnosis stability and progression": "the known diagnosis, whether it is stable or progressing, and any recent change",
    "Existing follow-up / referral status": "whether the patient is already under HES, has an appointment or has an active referral",
    "Current treatment and self-care tried": "current drops, lubricants, lid hygiene, spectacles or optometry treatment already tried",
    "PVD symptom onset and progression": "onset, duration, laterality and whether flashes/floaters are changing",
    "Dilated retinal examination / peripheral view": "dilated peripheral retinal examination findings, including widefield imaging or indentation if done",
    "Shafer sign / vitreous haemorrhage status": "whether there is Shafer sign/tobacco dust, pigment cells or vitreous haemorrhage",
    "Retinal risk history": "previous retinal detachment/tear/laser, high myopia, lattice, aphakia or trauma history",
    "Retinal tear treatment details": "retinal tear/hole treatment date, eye, treatment type and follow-up plan",
    "Referral identifiers and dates": "referral ID, date sent, pathway/provider, urgency and confirmation received",
    "Existing HES team and appointment status": "responsible HES team, last seen date, planned follow-up and next appointment status",
    "Patient contact and access needs": "best contact details and any travel, accessibility or appointment access needs",
    "Clinical change since referral / while waiting": "whether vision, pain, symptoms or other concerns have changed while waiting",
    "Post-cataract symptom negation / red-flag status": "whether there is pain, redness, light sensitivity, discharge, reduced vision or inflammation after cataract surgery",
    "Post-cataract operative details / discharge information": "operation date, eye, provider, complications, lens details, discharge letter and post-op drops",
    "Post-cataract OCT / CMO status": "OCT findings, whether cystoid macular oedema/fluid is present, how vision is affected and whether retina/HES are already monitoring",
    "Disc appearance, laterality and images / OCT comparison": "which eye, disc appearance, disc photos/OCT and any comparison with previous images",
    "Papilloedema symptom screen": "whether there are headaches, transient visual obscurations, pulsatile tinnitus, vomiting or double vision",
    "Neuro-ophthalmic visual function": "colour vision, visual fields, pupils and any neuro-ophthalmic symptoms",
    "Previous HES optic nerve review and stability": "previous HES optic nerve review outcome and whether the appearance is stable",
    "Iris lesion description and image": "iris lesion photo, size, location, colour, elevation, vessels and whether it is changing",
    "Anterior segment safety screen": "VA, IOP, pupil, anterior chamber/angle findings and whether there is pain, redness, photophobia, hyphaema or rubeosis",
    "Retinal lesion image, size, elevation and change": "photo/OCT, lesion size, location, whether it is flat/elevated, symptoms and whether it is new or changing",
    "Watering severity, TEAR score and intervention preference": "watering severity, laterality, duration, lid/ocular surface findings and whether the patient would consider intervention",
    "Corneal findings, contact lens status and refraction advice context": "corneal appearance, staining, symptoms, contact lens wear, VA/refraction and whether new glasses should be avoided",
    "Macular OCT images, VA, symptoms and change over time": "OCT images, VA, which eye, symptoms, duration, fundus findings and previous comparison scans",
    "Diabetic OCT, VA, retinopathy grade and previous treatment": "OCT/photos, VA, retinopathy or maculopathy grade, previous laser/injections and screening/HES status",
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


def openai_api_key():
    return st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")


def image_review_enabled():
    return bool(openai_api_key())


def content_type_for_upload(uploaded_file):
    if uploaded_file is None:
        return ""
    content_type = getattr(uploaded_file, "type", "") or ""
    if content_type:
        return content_type
    guessed, _encoding = mimetypes.guess_type(uploaded_file.name)
    return guessed or "application/octet-stream"


def image_review_error(filename, content_type, size_bytes, message):
    return {
        "attached": "Yes",
        "filename": filename or "",
        "content_type": content_type or "",
        "size_bytes": size_bytes or "",
        "model": IMAGE_REVIEW_MODEL,
        "prompt_version": IMAGE_REVIEW_PROMPT_VERSION,
        "image_accessible": "Unclear",
        "image_type": "Unclear",
        "image_quality": "Unclear",
        "finding_visible": "Unclear",
        "finding_category": "Unclear",
        "brief_image_finding": "",
        "supports_ag_request": "Unclear",
        "confidence": "Low",
        "limitations": message,
        "uncertainty_flag": "Yes - image review unavailable or uncertain",
        "error": message,
    }


def no_image_review():
    return {
        "attached": "No",
        "filename": "",
        "content_type": "",
        "size_bytes": "",
        "model": "",
        "prompt_version": "",
        "image_accessible": "",
        "image_type": "",
        "image_quality": "",
        "finding_visible": "",
        "finding_category": "",
        "brief_image_finding": "",
        "supports_ag_request": "",
        "confidence": "",
        "limitations": "",
        "uncertainty_flag": "",
        "error": "",
    }


def build_image_review_attachment(uploaded_file, data, content_type):
    encoded = base64.b64encode(data).decode("ascii")
    data_url = f"data:{content_type};base64,{encoded}"
    if content_type == "application/pdf":
        return {
            "type": "input_file",
            "filename": uploaded_file.name,
            "file_data": data_url,
        }
    return {"type": "input_image", "image_url": data_url, "detail": "high"}


def build_image_review_payload(question, uploaded_file, data, content_type):
    attachment = build_image_review_attachment(uploaded_file, data, content_type)
    user_text = (
        "A&G request / initial message:\n"
        f"{question}\n\n"
        "Please review the attached image/file and complete the structured image review labels."
    )
    return {
        "model": IMAGE_REVIEW_MODEL,
        "store": False,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": IMAGE_REVIEW_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}, attachment],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ag_image_review",
                "strict": True,
                "schema": IMAGE_REVIEW_SCHEMA,
            }
        },
    }


def extract_response_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("Could not find text output in image review response")


def call_openai_image_review(payload):
    request = Request(
        OPENAI_RESPONSES_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {openai_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            api_response = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        body = re.sub(r"sk-[A-Za-z0-9_\\-]+", "sk-REDACTED", body)
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI API network error: {exc}") from exc

    return json.loads(extract_response_text(api_response))


def image_uncertainty_flag(review):
    flags = []
    if review.get("confidence") in {"Low", "Medium"}:
        flags.append(f"{review.get('confidence')} confidence")
    if review.get("supports_ag_request") in {"Partly", "Unclear", "No"}:
        flags.append(f"relevance {review.get('supports_ag_request')}")
    if review.get("image_accessible") != "Yes":
        flags.append(f"image accessible {review.get('image_accessible')}")
    if review.get("image_quality") in {"Poor", "Unclear"}:
        flags.append(f"quality {review.get('image_quality')}")
    if review.get("finding_visible") in {"Partly", "No", "Unclear"}:
        flags.append(f"finding visible {review.get('finding_visible')}")
    return "Yes - " + "; ".join(flags) if flags else "No"


def analyse_uploaded_image(question, uploaded_file):
    if uploaded_file is None:
        return no_image_review()

    content_type = content_type_for_upload(uploaded_file)
    data = uploaded_file.getvalue()
    if content_type not in ALLOWED_IMAGE_REVIEW_TYPES:
        return image_review_error(
            uploaded_file.name,
            content_type,
            len(data),
            "Unsupported attachment type. Please use an image or PDF; video is excluded from this validation scope.",
        )
    if len(data) > IMAGE_REVIEW_MAX_BYTES:
        return image_review_error(
            uploaded_file.name,
            content_type,
            len(data),
            "Attachment is larger than the configured image-review limit.",
        )
    if not openai_api_key():
        return image_review_error(
            uploaded_file.name,
            content_type,
            len(data),
            "OPENAI_API_KEY is not configured, so image review was not run.",
        )

    payload = build_image_review_payload(question, uploaded_file, data, content_type)
    try:
        review = call_openai_image_review(payload)
    except Exception as exc:
        return image_review_error(uploaded_file.name, content_type, len(data), str(exc))

    review.update({
        "attached": "Yes",
        "filename": uploaded_file.name,
        "content_type": content_type,
        "size_bytes": len(data),
        "model": IMAGE_REVIEW_MODEL,
        "prompt_version": IMAGE_REVIEW_PROMPT_VERSION,
        "uncertainty_flag": image_uncertainty_flag(review),
        "error": "",
    })
    return review


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


def image_review_from_result(result):
    return result.get("Image Review", {}) or no_image_review()


def image_review_log_values(result, feedback=None):
    feedback = feedback or {}
    review = image_review_from_result(result)
    return [
        review.get("attached", ""),
        review.get("filename", ""),
        review.get("content_type", ""),
        review.get("size_bytes", ""),
        review.get("model", ""),
        review.get("prompt_version", ""),
        review.get("image_accessible", ""),
        review.get("image_type", ""),
        review.get("image_quality", ""),
        review.get("finding_visible", ""),
        review.get("finding_category", ""),
        review.get("brief_image_finding", ""),
        review.get("supports_ag_request", ""),
        review.get("confidence", ""),
        review.get("uncertainty_flag", ""),
        review.get("limitations", ""),
        review.get("error", ""),
        feedback.get("clinician_image_assessment", ""),
        feedback.get("clinician_image_notes", ""),
    ]


def plain_missing_information(text):
    text = str(text or "").strip()
    if not text:
        return ""
    if text in PLAIN_MISSING_INFO_LABELS:
        return PLAIN_MISSING_INFO_LABELS[text]
    cleaned = text
    replacements = {
        "symptom negation / red-flag status": "whether concerning symptoms are present or absent",
        "red-flag status": "whether any concerning symptoms are present",
        "red flags": "concerning symptoms",
        "red-flag": "concerning",
        "positive and negative symptoms": "symptoms that are present and symptoms that are absent",
        "status": "details",
        "Laterality": "which eye is affected",
        "VA": "vision/VA",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned[:1].lower() + cleaned[1:] if cleaned else cleaned


def plain_missing_information_list(items):
    values = []
    for item in items:
        value = plain_missing_information(item.get("Missing Information", ""))
        if value and value not in values:
            values.append(value)
    return values


def join_plain_request_items(items):
    items = [str(item).strip().rstrip(".") for item in items if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def plainify_request_details(details):
    text = str(details or "").strip().rstrip(".?")
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) > 1:
        plain = join_plain_request_items(plain_missing_information(part) for part in parts)
    else:
        plain = plain_missing_information(text)
    plain = plain.replace(
        "OCT findings, whether cystoid macular oedema/fluid is present, how vision is affected and whether retina/HES are already monitoring and whether there is pain, redness, light sensitivity, discharge, reduced vision or inflammation after cataract surgery",
        "the OCT findings, whether there is cystoid macular oedema/fluid, how vision is affected, whether retina clinic or HES are already monitoring, and whether there is any pain, redness, light sensitivity, discharge or inflammation after cataract surgery",
    )
    plain = plain.replace("retina/HES", "retina clinic or HES")
    return plain


def clinician_review_flags(result):
    if result.get("Review Override") == "glaucoma_adherence_advice":
        return "No", "Glaucoma adherence support advice", "Not required", "Existing pathway"
    if result.get("Review Override") == "suspected_preseptal_cellulitis_more_info":
        return "Optional", "Suspected eyelid cellulitis query needs basic history and examination findings", "Awaiting referrer information", "Existing pathway"

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
        info = join_plain_request_items(plain_missing_information_list(missing_info))
        if not info:
            info = "the clinical question, which eye is affected, vision/VA, symptom duration, symptoms present and absent, examination findings and any images/OCT/photos where available"
        result["Draft Response"] = {
            "Summary": "More information is needed before safe advice can be given.",
            "Suggested response": f"Thanks for this. I can advise more safely with a little more detail. Could you send {info}?",
            "Safety net": "If there are urgent symptoms or red flags, please refer urgently using the local ophthalmology pathway rather than waiting for advice.",
        }
    elif outcome.get("Outcome ID") == "OUT003":
        result["Draft Response"] = {
            "Summary": "The information provided suggests ophthalmology assessment is needed.",
            "Suggested response": (
                "Thanks for this. From the details provided, we will convert this A&G request into an ophthalmology referral. "
                "For future A&G requests, it is helpful to include symptom onset, VA, laterality, key positive and negative symptoms, relevant examination findings and any images/OCT/photos available."
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
        ("preseptal" in query or "pre septal" in query or "pre ceptal" in query or "cellulitis" in query)
        and ("lid" in query or "eyelid" in query or "swollen" in query or "swelling" in query)
    ):
        draft["Suggested response"] = (
            "Thanks for this. Could you send any relevant previous eye or medical history and the basic examination findings: "
            "how long the eyelid has been swollen, whether the child has a fever or seems unwell, vision if possible, "
            "eye movements, whether eye movement is painful, whether the eye is pushed forward, pupil findings, red eye, "
            "and a photo if available?"
        )
        draft["Safety net"] = (
            "If the child is systemically unwell, has reduced vision, painful or restricted eye movements, the eye is pushed forward, "
            "or there is severe headache, vomiting or concern about orbital cellulitis, please use the urgent same-day local pathway."
        )
        result["Review Override"] = "suspected_preseptal_cellulitis_more_info"
        result["Draft Response"] = draft
        return result

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
            "Thanks for this. The raised IOP and suspicious disc findings need glaucoma assessment, so we will convert this A&G request into a glaucoma referral. "
            "For future A&G requests, it is helpful to include IOP values and method, disc/OCT RNFL images, visual-field printouts/reliability, previous comparison and whether the patient is already under HES/glaucoma follow-up."
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
            "Thanks for this. From the details provided, we will convert this A&G request into an ophthalmology referral. "
            "For future A&G requests, it is helpful to include the symptom history, VA, laterality, relevant positive and negative findings, and any images/OCT/photos available."
        )

    if response.startswith("Please provide:"):
        details = response.replace("Please provide:", "", 1).strip().rstrip(".")
        if details:
            details = plainify_request_details(details)
            draft["Suggested response"] = (
                "Thanks for this. I can advise more safely with a little more detail. "
                f"Could you send {details}"
            )
            if not draft["Suggested response"].endswith((".", "?")):
                draft["Suggested response"] += "?"

    elif response.startswith("Please provide "):
        details = response.replace("Please provide ", "", 1).strip().rstrip(".")
        if details:
            details = plainify_request_details(details)
            draft["Suggested response"] = (
                "Thanks for this. I can advise more safely with a little more detail. "
                f"Could you send {details}"
            )
            if not draft["Suggested response"].endswith((".", "?")):
                draft["Suggested response"] += "?"

    response = str(draft.get("Suggested response", "") or "").strip()
    if "Could you send " in response:
        prefix, details = response.split("Could you send ", 1)
        draft["Suggested response"] = f"{prefix}Could you send {plainify_request_details(details)}?"

    draft["Suggested response"] = (
        str(draft.get("Suggested response", "") or "")
        .replace("symptom negation / red-flag status", "whether concerning symptoms are present or absent")
        .replace("red-flag status", "whether any concerning symptoms are present")
        .replace("red flags", "concerning symptoms")
        .replace("positive and negative symptoms", "symptoms that are present and symptoms that are absent")
        .replace("laterality", "which eye is affected")
        .replace("Laterality", "which eye is affected")
        .replace("Amsler status", "Amsler/distortion symptoms")
        .replace("distortion/Amsler status", "distortion or Amsler symptoms")
        .replace("VA impact", "how vision is affected")
        .replace("pseudophakia status", "lens implant/pseudophakia details")
        .replace("symptom onset/change", "when symptoms started and whether they are changing")
    )

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
        "suggestion_helpful": "",
        "clinician_image_assessment": "",
        "clinician_image_notes": "",
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
    confidence, confidence_reason = suggested_response_confidence(result)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        needs_review,
        review_reason,
        review_status,
        reviewer_category,
        "",
        feedback.get("clinician_final_response", ""),
        feedback.get("clinician_final_outcome", ""),
        learning["learning_candidate"],
        learning["proposed_new_feature"],
        learning["proposed_new_presentation"],
        learning["proposed_missing_info"],
        learning["proposed_safety_condition"],
        learning["proposed_validation_case"],
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        outcome.get("Rationale", ""),
        confidence,
        confidence_reason,
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
        feedback.get("suggestion_helpful", ""),
        "Yes" if feedback.get("reasoning_not_satisfactory") else "No",
        feedback.get("tool_use_status", ""),
        feedback.get("override_reason", ""),
        learning["learning_candidate"],
        learning["learning_status"],
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        draft.get("Suggested response", ""),
    ]
    return row + image_review_log_values(result, feedback)


def result_to_review_queue_row(question, result, feedback=None):
    feedback = feedback or empty_feedback()
    learning = graph_learning_payload(question, result, feedback)
    outcome = result.get("Outcome Recommendation", {})
    presentations = result.get("Presentation Ranking", [])
    top_presentation = presentations[0] if presentations else {}
    draft = result.get("Draft Response", {})
    needs_review, review_reason, review_status, reviewer_category = clinician_review_flags(result)
    confidence, confidence_reason = suggested_response_confidence(result)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        needs_review,
        review_reason,
        review_status,
        reviewer_category,
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        confidence,
        confidence_reason,
        top_presentation.get("Presentation ID", ""),
        top_presentation.get("Presentation", ""),
        join_values(result.get("Detected Features", []), "Feature ID"),
        join_values(result.get("Detected Features", []), "Feature"),
        join_values(result.get("Missing Information", []), "Missing Information"),
        draft.get("Suggested response", ""),
        "",
        feedback.get("clinician_final_response", ""),
        feedback.get("clinician_final_outcome", ""),
        feedback.get("suggestion_helpful", ""),
        feedback.get("override_reason", ""),
        learning["proposed_change_type"] or "",
    ]
    return row + image_review_log_values(result, feedback)


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
    confidence, confidence_reason = suggested_response_confidence(result)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        outcome.get("Outcome ID", ""),
        outcome.get("Outcome", ""),
        outcome.get("Rationale", ""),
        confidence,
        confidence_reason,
        draft.get("Suggested response", ""),
        feedback.get("clinician_final_outcome", ""),
        feedback.get("clinician_final_response", ""),
        feedback.get("suggestion_helpful", ""),
        "Yes" if feedback.get("reasoning_not_satisfactory") else "No",
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
    return row + image_review_log_values(result, feedback)


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


def diagnostic_row(headers, values):
    row = [""] * len(headers)
    for column, value in values.items():
        if column in headers:
            row[headers.index(column)] = value
    return row


def run_logging_diagnostic():
    if not st.secrets.get("GOOGLE_APPS_SCRIPT_URL", ""):
        return "Google Apps Script URL is not configured in Streamlit secrets."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_with_apps_script([
        {
            "name": "A&G Log",
            "headers": SHEET_COLUMNS,
            "row": diagnostic_log_row(),
        },
        {
            "name": "Clinician Review Queue",
            "headers": REVIEW_QUEUE_COLUMNS,
            "row": diagnostic_row(REVIEW_QUEUE_COLUMNS, {
                "Timestamp": timestamp,
                "Question": "DIAGNOSTIC TEST - review queue logging",
                "Needs clinician review": "Yes",
                "Review reason": "Diagnostic test row",
                "Review status": "Diagnostic",
                "Reviewer category": "System test",
                "Suggested response confidence": "Low",
                "Suggested response confidence reason": "Diagnostic test row",
                "Clinician response": "If this row appears, review queue logging works.",
                "Clinician outcome": "TEST",
                "Suggested response helpful": "Diagnostic",
                "Reviewer notes": "Diagnostic test",
                "Graph update needed": "Diagnostic",
            }),
        },
        {
            "name": "Graph Update Candidates",
            "headers": GRAPH_CANDIDATE_COLUMNS,
            "row": diagnostic_row(GRAPH_CANDIDATE_COLUMNS, {
                "Candidate timestamp": timestamp,
                "Source question": "DIAGNOSTIC TEST - graph candidate logging",
                "Candidate status": "Diagnostic",
                "Candidate reason": "Diagnostic test row",
                "Reviewer category": "System test",
                "Proposed change type": "Diagnostic",
                "Clinician response": "If this row appears, graph candidate logging works.",
            }),
        },
        {
            "name": "Clinician Feedback",
            "headers": FEEDBACK_COLUMNS,
            "row": diagnostic_row(FEEDBACK_COLUMNS, {
                "Timestamp": timestamp,
                "Question": "DIAGNOSTIC TEST - clinician feedback logging",
                "Tool suggested outcome ID": "TEST",
                "Tool suggested outcome": "Diagnostic logging test",
                "Suggested response confidence": "High",
                "Suggested response confidence reason": "Diagnostic test row",
                "Tool suggested draft response": "If this row appears, clinician feedback logging works.",
                "Clinician final outcome": "TEST",
                "Clinician final response": "Diagnostic test",
                "Suggested response helpful": "Diagnostic",
                "Clinical reasoning concern": "No",
                "Tool use status": "Diagnostic",
            }),
        },
    ])
    return "Diagnostic rows sent to all Google Sheet tabs."


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


def render_image_review(review):
    if not review or review.get("attached") != "Yes":
        return

    st.subheader("Image summary")
    if review.get("error"):
        st.warning(f"Image review was not completed: {review.get('limitations') or review.get('error')}")
        return

    flag = review.get("uncertainty_flag", "")
    if flag and flag != "No":
        st.warning(f"Image output needs clinician attention: {flag}")
    else:
        st.success("Image output did not trigger an uncertainty flag.")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Image type", review.get("image_type", ""))
        col2.metric("Confidence", review.get("confidence", ""))
        col3.metric("Relevant to A&G", review.get("supports_ag_request", ""))

        st.markdown("**Visible finding**")
        st.write(review.get("brief_image_finding", ""))

        detail_cols = st.columns(3)
        detail_cols[0].write(f"**Quality:** {review.get('image_quality', '')}")
        detail_cols[1].write(f"**Finding visible:** {review.get('finding_visible', '')}")
        detail_cols[2].write(f"**Finding category:** {review.get('finding_category', '')}")

        limitations = review.get("limitations", "")
        if limitations:
            st.caption(f"Limitations: {limitations}")
        st.caption(
            "This is clinician-support information only. The clinician remains responsible for final image interpretation and the A&G response."
        )


def render_analysis_result(result):
    render_clinician_review_notice(result)
    outcome = result.get("Outcome Recommendation", {})
    draft = result.get("Draft Response", {})

    st.subheader("Suggested outcome")
    render_outcome(outcome)

    render_image_review(image_review_from_result(result))

    st.subheader("Draft response")
    render_draft_response(draft)

    with st.expander("Detected features and safety checks"):
        st.markdown("**Detected features**")
        render_features(result.get("Detected Features", []))
        st.markdown("**Safety checks**")
        render_safety(result.get("Safety Ranking", []))
        st.markdown("**Missing information**")
        render_missing_info(result.get("Missing Information", []))


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
        clinician_image_assessment = ""
        clinician_image_notes = ""
        if image_review_from_result(result).get("attached") == "Yes":
            st.markdown("**Image summary review**")
            clinician_image_assessment = st.radio(
                "How did you use the GPT image summary?",
                [
                    "Accepted",
                    "Used with edits / caution",
                    "Not used",
                    "Image summary unavailable",
                ],
                horizontal=True,
            )
            clinician_image_notes = st.text_area(
                "Image summary notes",
                value="",
                height=80,
                placeholder="Optional: note what was corrected, ignored, or clinically important.",
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
        "suggestion_helpful": "No - needs improvement" if suggestion_helpful in ("👎 Needs improvement", "Needs improvement") else "Yes - helpful",
        "clinician_image_assessment": clinician_image_assessment,
        "clinician_image_notes": clinician_image_notes.strip(),
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
    uploaded_image = st.file_uploader(
        "Optional image/PDF attachment",
        type=["png", "jpg", "jpeg", "webp", "gif", "pdf"],
        help="Attach an ocular image or PDF if available. Video is outside the current validated image/PDF scope.",
    )
    if uploaded_image is not None and not image_review_enabled():
        st.info("Image upload is available, but GPT image review needs OPENAI_API_KEY in Streamlit secrets or the environment.")

    analyse = st.button("Create suggested response", type="primary")

    if analyse:
        cleaned = question.strip()
        if not cleaned:
            st.warning("Enter an A&G question first.")
            return

        result = engine.analyse(cleaned)
        result = prepare_result_for_display(cleaned, result)
        if uploaded_image is not None:
            with st.spinner("Reviewing attached image/PDF..."):
                result["Image Review"] = analyse_uploaded_image(cleaned, uploaded_image)
        else:
            result["Image Review"] = no_image_review()
        st.session_state["last_question"] = cleaned
        st.session_state["last_result"] = result

    if st.session_state.get("last_result"):
        cleaned = st.session_state["last_question"]
        result = st.session_state["last_result"]
        render_analysis_result(result)
        render_clinician_feedback_form(cleaned, result)


if __name__ == "__main__":
    main()
