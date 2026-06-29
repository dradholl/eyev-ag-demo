"""Streamlit demo app for the EyeV A&G reasoning engine."""

from pathlib import Path

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
    st.caption("V7.13.2 prototype. Demo use only. Do not enter patient-identifiable information unless you have local approval.")

    with st.sidebar:
        st.header("Examples")
        selected_example = st.selectbox("Choose a test question", [""] + EXAMPLES)
        st.divider()
        st.subheader("Deployment note")
        st.write("For public demo links, set an app password and use synthetic or anonymised cases only.")

    default_text = selected_example or EXAMPLES[0]
    question = st.text_area("A&G request text", value=default_text, height=180)

    analyse = st.button("Analyse", type="primary")

    if analyse:
        cleaned = question.strip()
        if not cleaned:
            st.warning("Enter an A&G question first.")
            return

        result = engine.analyse(cleaned)

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


if __name__ == "__main__":
    main()
