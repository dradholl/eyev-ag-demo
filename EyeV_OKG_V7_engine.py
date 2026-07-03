"""EyeV OKG V7.0 locked clinical reasoning engine.

This engine reads EyeV_Ophthalmic_Knowledge_Graph_v2.xlsx and produces
traceable Advice & Guidance reasoning output. It supports clinician reasoning;
it is not an autonomous diagnostic tool.
"""

# Cell 1 — Imports and workbook loading

import pandas as pd

import re
from datetime import datetime
from pathlib import Path

DEFAULT_GRAPH_FILE = "EyeV_Ophthalmic_Knowledge_Graph_v2.xlsx"



# Cell 2 — OKGEngine V7.0 locked class

class OKGEngine:
    """Workbook-driven ophthalmic clinical reasoning engine."""

    REQUIRED_SHEETS = [
        "Entities",
        "Relationships",
        "Rules",
        "Validation",
        "Inference_Policies",
    ]

    NEGATION_CUES = (
        "no",
        "not",
        "without",
        "denies",
        "denied",
        "absence of",
        "nil",
    )

    def __init__(self, graph_file=None):
        self.graph_file = self._resolve_graph_file(graph_file)
        self.entities = pd.read_excel(self.graph_file, sheet_name="Entities", dtype=str).fillna("")
        self.relationships = pd.read_excel(self.graph_file, sheet_name="Relationships", dtype=str).fillna("")
        self.rules = pd.read_excel(self.graph_file, sheet_name="Rules", dtype=str).fillna("")
        self.validation = pd.read_excel(self.graph_file, sheet_name="Validation", dtype=str).fillna("")
        self.policies = pd.read_excel(self.graph_file, sheet_name="Inference_Policies", dtype=str).fillna("")

        try:
            self.outcome_mapping = pd.read_excel(self.graph_file, sheet_name="Outcome_Mapping", dtype=str).fillna("")
        except Exception:
            self.outcome_mapping = pd.DataFrame()

        for df in [
            self.entities,
            self.relationships,
            self.rules,
            self.validation,
            self.policies,
            self.outcome_mapping,
        ]:
            if len(df.columns):
                df.columns = [str(c).strip() for c in df.columns]

    def _resolve_graph_file(self, graph_file):
        if graph_file:
            return str(graph_file)

        local = Path(DEFAULT_GRAPH_FILE)
        if local.exists():
            return str(local)

        notebook_dir_file = Path.cwd() / DEFAULT_GRAPH_FILE
        if notebook_dir_file.exists():
            return str(notebook_dir_file)

        try:
            from google.colab import files
            print("Upload:", DEFAULT_GRAPH_FILE)
            uploaded = files.upload()
            candidates = [name for name in uploaded if name.endswith(".xlsx")]
            exact = [name for name in candidates if name == DEFAULT_GRAPH_FILE]
            if exact:
                return exact[0]
            okg = [name for name in candidates if "knowledge_graph" in name.lower() or "okg" in name.lower()]
            if okg:
                return okg[0]
        except Exception:
            pass

        raise FileNotFoundError(
            f"Could not find {DEFAULT_GRAPH_FILE}. Pass graph_file=... or place the workbook beside this notebook."
        )

    def normalise(self, text):
        text = str(text).lower()
        replacements = {
            "loss of vision": "vision loss",
            "loss of sight": "vision loss",
            "lost vision": "vision loss",
            "lost sight": "vision loss",
            "right eye": "one eye",
            "left eye": "one eye",
            "r eye": "one eye",
            "l eye": "one eye",
            "can't": "cannot",
            "cant": "cannot",
            "couldn't": "could not",
        }
        for old, new in replacements.items():
            if old in {"r eye", "l eye"}:
                text = re.sub(rf"\b{re.escape(old)}\b", new, text)
            else:
                text = text.replace(old, new)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_negated(self, normalised_text, term):
        q = f" {normalised_text} "
        term = self.normalise(term)
        if not term or term not in q:
            return False

        term_pattern = re.escape(term).replace(r"\ ", r"\s+")
        for cue in self.NEGATION_CUES:
            cue_pattern = re.escape(cue).replace(r"\ ", r"\s+")
            before = rf"\b{cue_pattern}\b(?:\s+\w+){{0,5}}\s+{term_pattern}\b"
            if re.search(before, q):
                return True
        return False

    def detect_features(self, query):
        q = self.normalise(query)
        detected = []
        detected_ids = set()

        features = self.entities[self.entities["Entity Type"] == "Feature"]
        for _, feature in features.iterrows():
            feature_id = feature["Entity ID"]
            terms = []
            for column in ["Entity Name", "Synonyms / Phrases", "Normalised Terms"]:
                value = str(feature.get(column, ""))
                terms.extend([part.strip() for part in value.split(";") if part.strip()])

            matched_terms = []
            for term in terms:
                normalised_term = self.normalise(term)
                if len(normalised_term) < 4:
                    continue
                if normalised_term and normalised_term in q and not self.is_negated(q, normalised_term):
                    if feature_id == "OF003" and self._monocular_term_is_diplopia_context(q):
                        continue
                    if feature_id == "OF002" and "transient visual obscurations" in q:
                        continue
                    if feature_id == "OF006" and self._field_term_is_glaucoma_context(q, normalised_term):
                        continue
                    if feature_id == "OF008" and self._oct_term_is_glaucoma_context(q):
                        continue
                    if feature_id == "OF012" and "optic disc drusen" in q:
                        continue
                    if feature_id == "OF016" and self._visual_fields_are_missing_context(q):
                        continue
                    if feature_id == "OF018" and self._papilloedema_symptom_context(q) and "red eye" not in q and "halos" not in q:
                        continue
                    if feature_id == "OF021" and self._contact_lens_is_negated(q):
                        continue
                    if feature_id == "OF032" and not self._cataract_context(q):
                        continue
                    if feature_id == "OF024" and self._watery_eye_is_lid_context(q):
                        continue
                    if feature_id == "OF027" and not self._cataract_context(q):
                        continue
                    if feature_id == "OF031" and self._post_op_red_flags_are_negated(q):
                        continue
                    if feature_id == "OF042" and not self._diabetic_context(q):
                        continue
                    if feature_id == "OF043" and not self._diabetic_context(q):
                        continue
                    if feature_id == "OF044" and not self._diabetic_context(q):
                        continue
                    if feature_id == "OF057" and self._diabetic_context(q):
                        continue
                    if feature_id == "OF038" and not self._orbital_context(q):
                        continue
                    if feature_id == "OF046" and self._orbital_context(q):
                        continue
                    if feature_id == "OF047" and self._orbital_context(q):
                        continue
                    if feature_id == "OF048" and not self._adult_diplopia_or_neuro_context(q):
                        continue
                    if feature_id == "OF069" and not self._optic_disc_swelling_context(q):
                        continue
                    if feature_id == "OF070" and not self._papilloedema_symptom_context(q):
                        continue
                    if feature_id == "OF072" and not self._neuro_ophthalmology_context(q):
                        continue
                    if feature_id == "OF074" and not self._iris_anterior_segment_context(q):
                        continue
                    if feature_id == "OF075" and not self._iris_red_flag_context(q):
                        continue
                    matched_terms.append(term)

            if matched_terms:
                detected.append({
                    "Feature ID": feature_id,
                    "Feature": feature["Entity Name"],
                    "Matched Terms": sorted(set(matched_terms)),
                })
                detected_ids.add(feature_id)

        fallback_patterns = {
            "OF001": lambda text: "sudden" in text and "vision loss" in text,
            "OF002": lambda text: (
                "transient" in text
                and "transient visual obscurations" not in text
                or "temporary" in text
                or "vision returned" in text
                or "vision recovered" in text
                or ("vision loss" in text and "now normal" in text)
            ),
            "OF003": lambda text: (
                ("one eye" in text or "monocular" in text or "unilateral" in text)
                and not self._monocular_term_is_diplopia_context(text)
            ),
            "OF004": lambda text: ("flashes" in text or "photopsia" in text) and not self.is_negated(text, "flashes"),
            "OF005": lambda text: ("floaters" in text or "cobwebs" in text) and not self.is_negated(text, "floaters"),
            "OF006": lambda text: (
                ("curtain" in text or "field loss" in text or "field defect" in text or "shadow" in text)
                and not self.is_negated(text, "curtain")
                and not self.is_negated(text, "field loss")
                and not self.is_negated(text, "field defect")
                and not self._field_term_is_glaucoma_context(text, "field loss")
                and not self._field_term_is_glaucoma_context(text, "field defect")
            ),
            "OF007": lambda text: (
                (
                    "painful red eye" in text
                    or "red painful eye" in text
                    or "ocular pain" in text
                    or "eye pain" in text
                    or ("painful" in text and "red eye" in text)
                )
                and not self.is_negated(text, "painful red eye")
                and not self.is_negated(text, "red painful eye")
                and not self.is_negated(text, "ocular pain")
                and not self.is_negated(text, "eye pain")
            ),
            "OF008": lambda text: (
                "oct" in text
                and "not attached" not in text
                and "no oct" not in text
                and "without oct" not in text
                and not self._oct_term_is_glaucoma_context(text)
            ),
            "OF009": lambda text: ("macula" in text or "macular" in text) and not self.is_negated(text, "macula"),
            "OF010": lambda text: (
                (
                    "distortion" in text
                    or "distorted vision" in text
                    or "wavy lines" in text
                    or "metamorphopsia" in text
                    or "amsler" in text
                )
                and not self.is_negated(text, "distortion")
                and not self.is_negated(text, "distorted vision")
                and not self.is_negated(text, "wavy lines")
                and not self.is_negated(text, "metamorphopsia")
                and not self.is_negated(text, "amsler")
            ),
            "OF011": lambda text: (
                (
                    "wet amd" in text
                    or "cnv" in text
                    or "subretinal fluid" in text
                    or "intraretinal fluid" in text
                    or "macular fluid" in text
                    or "fluid on oct" in text
                )
                and not self.is_negated(text, "wet amd")
                and not self.is_negated(text, "cnv")
                and not self.is_negated(text, "fluid")
            ),
            "OF012": lambda text: (
                (
                    "dry amd" in text
                    or "drusen" in text
                    or "non exudative amd" in text
                    or "amd monitoring" in text
                )
                and "optic disc drusen" not in text
            ),
            "OF013": lambda text: (
                (
                    "reduced central vision" in text
                    or "central blur" in text
                    or "central blurred vision" in text
                    or "blurred central vision" in text
                    or "va reduced" in text
                    or "visual acuity reduced" in text
                    or "reduced vision" in text
                    or "vision worse" in text
                    or "worsening vision" in text
                    or "decreased vision" in text
                )
                and not self.is_negated(text, "reduced central vision")
                and not self.is_negated(text, "central blur")
                and not self.is_negated(text, "central blurred vision")
                and not self.is_negated(text, "blurred central vision")
                and not self.is_negated(text, "va reduced")
                and not self.is_negated(text, "visual acuity reduced")
                and not self.is_negated(text, "reduced vision")
                and not self.is_negated(text, "decreased vision")
            ),
            "OF014": lambda text: (
                "raised iop" in text
                or "high iop" in text
                or "iop raised" in text
                or "ocular hypertension" in text
                or "pressure high" in text
                or "raised pressures" in text
                or "mmhg" in text
            ),
            "OF015": lambda text: (
                "optic disc" in text
                or "disc margin" in text
                or "disc margins" in text
                or "disc cupping" in text
                or "cupping" in text
                or "cup disc" in text
                or "suspicious disc" in text
                or "disc haemorrhage" in text
                or "disc hemorrhage" in text
                or "healthy discs" in text
                or "rnfl" in text
                or "oct rnfl" in text
            ),
            "OF016": lambda text: (
                (
                    "visual field" in text
                    or "visual fields" in text
                    or "field defect" in text
                    or "field loss" in text
                    or "fields worse" in text
                    or "field progression" in text
                    or "full visual fields" in text
                )
                and not self.is_negated(text, "field loss")
                and not self.is_negated(text, "field defect")
                and not self._visual_fields_are_missing_context(text)
            ),
            "OF017": lambda text: (
                "glaucoma suspect" in text
                or "suspect glaucoma" in text
                or "suspicious for glaucoma" in text
                or "glaucoma query" in text
                or "possible glaucoma" in text
                or "glaucoma referral" in text
                or "glaucoma monitoring" in text
                or "glaucoma history" in text
            ),
            "OF018": lambda text: (
                (
                    "angle closure" in text
                    or "narrow angles" in text
                    or "acute glaucoma" in text
                    or "halos" in text
                    or "nausea" in text
                    or "vomiting" in text
                    or "severe eye pain" in text
                    or "headache with red eye" in text
                    or "fixed pupil" in text
                )
                and not self.is_negated(text, "halos")
                and not self.is_negated(text, "nausea")
                and not self.is_negated(text, "painful red eye")
                and not self.is_negated(text, "severe eye pain")
                and not (self._papilloedema_symptom_context(text) and "red eye" not in text and "halos" not in text)
            ),
            "OF019": lambda text: (
                "glaucoma drops" in text
                or "eye drops" in text
                or "latanoprost" in text
                or "timolol" in text
                or "medication" in text
                or "compliance" in text
                or "intolerance" in text
                or "prostaglandin" in text
            ),
            "OF020": lambda text: (
                (
                    "photophobia" in text
                    or "photophobic" in text
                    or "light sensitivity" in text
                    or "light sensitive" in text
                    or "sensitive to light" in text
                )
                and not self.is_negated(text, "photophobia")
                and not self.is_negated(text, "photophobic")
                and not self.is_negated(text, "light sensitivity")
                and not self.is_negated(text, "light sensitive")
                and not self.is_negated(text, "sensitive to light")
            ),
            "OF021": lambda text: (
                (
                    "contact lens" in text
                    or "contact lenses" in text
                    or "contact lens wearer" in text
                    or "cl wearer" in text
                    or "lens wearer" in text
                )
                and not self.is_negated(text, "contact lens")
                and not self.is_negated(text, "contact lenses")
                and not self._contact_lens_is_negated(text)
            ),
            "OF022": lambda text: (
                (
                    "corneal abrasion" in text
                    or "corneal ulcer" in text
                    or "keratitis" in text
                    or "corneal infiltrate" in text
                    or "infiltrate" in text
                    or "corneal opacity" in text
                    or "corneal staining" in text
                )
                and not self.is_negated(text, "corneal staining")
                and not self.is_negated(text, "staining")
                and "no corneal staining details" not in text
                and "no staining details" not in text
            ),
            "OF023": lambda text: (
                (
                    "red eye" in text
                    or "red eyes" in text
                    or "eye redness" in text
                    or "red painful eye" in text
                    or "red painful" in text
                    or "conjunctival injection" in text
                )
                and not self.is_negated(text, "red eye")
            ),
            "OF024": lambda text: (
                (
                    "discharge" in text
                    or "sticky eye" in text
                    or "sticky eyes" in text
                    or "sticky lids" in text
                    or "conjunctivitis" in text
                    or "watery sticky eye" in text
                )
                and "watery eye from lid" not in text
                and "ectropion with watery eye" not in text
                and not self._watery_eye_is_lid_context(text)
            ),
            "OF025": lambda text: (
                "uveitis" in text
                or "iritis" in text
                or "anterior uveitis" in text
                or "cells" in text
                or "flare" in text
                or "anterior chamber reaction" in text
                or "ciliary flush" in text
            ),
            "OF026": lambda text: (
                "cataract" in text
                or "cataracts" in text
                or "lens opacity" in text
                or "cataract referral" in text
                or "cataract assessment" in text
            ),
            "OF027": lambda text: (
                self._cataract_context(text)
                and (
                    "glare" in text
                    or "night driving" in text
                    or "driving difficulty" in text
                    or "dazzle" in text
                    or "functional difficulty" in text
                    or "daily living" in text
                    or "daily activities" in text
                )
                and not self.is_negated(text, "glare")
                and not self.is_negated(text, "daily living")
            ),
            "OF028": lambda text: (
                (
                    "pco" in text
                    or "posterior capsular opacification" in text
                    or "posterior capsule opacification" in text
                    or "capsule opacity" in text
                    or "cloudy capsule" in text
                )
                and not self.is_negated(text, "pco")
                and not self.is_negated(text, "posterior capsular opacification")
                and not self.is_negated(text, "posterior capsule opacification")
            ),
            "OF029": lambda text: (
                (
                    "yag" in text
                    or "yag laser" in text
                    or "yag capsulotomy" in text
                    or "capsulotomy" in text
                    or "laser capsulotomy" in text
                )
                and not self.is_negated(text, "yag")
            ),
            "OF030": lambda text: (
                "post cataract surgery" in text
                or "cataract surgery" in text
                or "after cataract surgery" in text
                or "post op cataract" in text
                or "post cataract" in text
                or "post cat" in text
                or "post op sight test" in text
                or "pseudophakia" in text
                or "pseudophakic" in text
                or "lens implant" in text
            ),
            "OF031": lambda text: (
                (
                    "post op pain" in text
                    or "post operative pain" in text
                    or "pain after surgery" in text
                    or "red eye after surgery" in text
                    or "reduced vision after surgery" in text
                    or "painful eye after cataract surgery" in text
                    or ("post cataract" in text and ("poor vision" in text or "reduced vision" in text or "staining" in text))
                    or ("post cat" in text and ("poor vision" in text or "reduced vision" in text or "staining" in text))
                    or ("post op" in text and ("poor vision" in text or "reduced vision" in text or "staining" in text))
                    or (
                        "cataract surgery" in text
                        and ("pain" in text or "red eye" in text or "reduced vision" in text or "worsening vision" in text)
                    )
                )
                and not self.is_negated(text, "pain")
                and not self.is_negated(text, "red eye")
                and not self.is_negated(text, "reduced vision")
                and not self._post_op_red_flags_are_negated(text)
            ),
            "OF032": lambda text: (
                ("cataract" in text or "pseudophakia" in text or "pseudophakic" in text)
                and (
                    "no visual acuity" in text
                    or "no va" in text
                    or "no information about daily living" in text
                    or "no functional impact" in text
                    or "no laterality" in text
                    or "unclear visual potential" in text
                    or "ocular comorbidity" in text
                    or "comorbidity" in text
                    or "glaucoma history" in text
                )
            ),
            "OF033": lambda text: (
                "lid lesion" in text
                or "eyelid lesion" in text
                or "eyelid lump" in text
                or "lid lump" in text
                or "eyelid cyst" in text
                or "eyelid swelling" in text
                or "lump on eyelid" in text
                or "eyelid" in text and "lesion" in text
            ),
            "OF034": lambda text: (
                "chalazion" in text
                or "stye" in text
                or "hordeolum" in text
                or "meibomian cyst" in text
                or "lid cyst" in text
            ),
            "OF035": lambda text: (
                (
                    "growing lesion" in text
                    or "increasing in size" in text
                    or "increasing size" in text
                    or "ulcerated" in text
                    or "ulceration" in text
                    or "bleeding" in text
                    or "lash loss" in text
                    or "madarosis" in text
                    or "pigmented eyelid lesion" in text
                    or "pigmented lesion" in text
                    or "recurrent chalazion" in text
                    or "irregular lid margin" in text
                    or "crusting margin" in text
                )
                and not self.is_negated(text, "growth")
                and not self.is_negated(text, "lash loss")
                and not self.is_negated(text, "bleeding")
            ),
            "OF036": lambda text: (
                "ptosis" in text
                or "droopy lid" in text
                or "drooping eyelid" in text
                or "lid droop" in text
                or "eyelid droop" in text
            ),
            "OF037": lambda text: (
                "ectropion" in text
                or "entropion" in text
                or "lid malposition" in text
                or "inturned lashes" in text
                or "lashes rubbing" in text
                or "watery eye from lid" in text
            ),
            "OF038": lambda text: (
                (
                    "proptosis" in text
                    or "bulging eye" in text
                    or "orbital swelling" in text
                    or "painful eye movements" in text
                    or "restricted eye movements" in text
                    or "orbital signs" in text
                    or "diplopia with proptosis" in text
                )
                and not self.is_negated(text, "proptosis")
                and not self.is_negated(text, "diplopia")
                and self._orbital_context(text)
            ),
            "OF039": lambda text: (
                "diabetes" in text
                or "diabetic" in text
                or "diabetic eye screening" in text
                or "diabetic retinopathy" in text
            ),
            "OF040": lambda text: (
                (
                    "dmo" in text
                    or "diabetic macular oedema" in text
                    or "diabetic macular edema" in text
                    or "diabetic maculopathy" in text
                    or "macular oedema" in text
                    or "macular edema" in text
                    or "macular thickening" in text
                )
                and not self.is_negated(text, "dmo")
                and not self.is_negated(text, "diabetic macular oedema")
                and not self.is_negated(text, "diabetic macular edema")
                and not self.is_negated(text, "diabetic maculopathy")
            ),
            "OF041": lambda text: (
                "diabetic retinopathy" in text
                or "background diabetic retinopathy" in text
                or "background retinopathy" in text
                or "pre proliferative" in text
                or "preproliferative" in text
                or "proliferative diabetic retinopathy" in text
                or "pdr" in text
                or "r2" in text
                or "r3" in text
            ),
            "OF042": lambda text: (
                self._diabetic_context(text)
                and (
                    "new vessels" in text
                    or "neovascularisation" in text
                    or "neovascularization" in text
                    or "nvd" in text
                    or "nve" in text
                    or "vitreous haemorrhage" in text
                    or "vitreous hemorrhage" in text
                    or "preretinal haemorrhage" in text
                )
            ),
            "OF043": lambda text: (
                "prp" in text
                or "panretinal photocoagulation" in text
                or "retinal laser" in text
                or "focal laser" in text
                or "anti vegf" in text
                or "anti-vegf" in text
                or "intravitreal injections" in text
            ),
            "OF044": lambda text: (
                self._diabetic_context(text)
                and (
                    "stable diabetic retinopathy" in text
                    or "stable diabetic screening" in text
                    or "annual screening" in text
                    or "routine screening" in text
                    or "no progression" in text
                    or "stable changes" in text
                    or "vision unchanged" in text
                )
            ),
            "OF045": lambda text: (
                "squint" in text
                or "strabismus" in text
                or "eye turn" in text
                or "turned eye" in text
                or "ocular alignment" in text
                or "esotropia" in text
                or "exotropia" in text
            ),
            "OF046": lambda text: (
                (
                    "diplopia" in text
                    or "double vision" in text
                    or "seeing double" in text
                    or "binocular diplopia" in text
                    or "monocular diplopia" in text
                )
                and not self.is_negated(text, "diplopia")
                and not self.is_negated(text, "double vision")
                and not self._orbital_context(text)
            ),
            "OF047": lambda text: (
                (
                    "restricted eye movements" in text
                    or "ocular motility" in text
                    or "motility defect" in text
                    or "sixth nerve palsy" in text
                    or "third nerve palsy" in text
                    or "fourth nerve palsy" in text
                    or "nerve palsy" in text
                )
                and not self.is_negated(text, "restricted eye movements")
                and not self.is_negated(text, "ocular motility")
                and not self.is_negated(text, "motility defect")
                and not self._orbital_context(text)
            ),
            "OF048": lambda text: self._adult_diplopia_or_neuro_context(text),
            "OF049": lambda text: (
                "child" in text
                or "paediatric" in text
                or "pediatric" in text
                or "amblyopia" in text
                or "lazy eye" in text
                or "child squint" in text
                or "paediatric squint" in text
            ),
            "OF050": lambda text: (
                "low vision" in text
                or "sight impairment" in text
                or "sight impaired" in text
                or "cvi" in text
                or "certificate of vision impairment" in text
                or "visual rehabilitation" in text
                or "low vision clinic" in text
            ),
            "OF051": lambda text: (
                "stable" in text
                or "no change" in text
                or "routine monitoring" in text
                or "routine review" in text
                or "monitoring due" in text
                or "follow up due" in text
                or "unchanged vision" in text
                or "no new symptoms" in text
                or "no clinical change" in text
            ),
            "OF052": lambda text: (
                "chasing referral" in text
                or "chasing appointment" in text
                or "hes appointment" in text
                or "hospital appointment" in text
                or "previous referral" in text
                or "referral status" in text
                or "referral id" in text
                or "not triaged" in text
                or "awaiting triage" in text
                or "not heard anything" in text
                or "appointment history" in text
                or "clinic appointment history" in text
                or "previous appointment" in text
            ),
            "OF053": lambda text: (
                "refraction" in text
                or "refractive change" in text
                or "glasses" in text
                or "spectacles" in text
                or "prescription" in text
                or "optician" in text
                or "optometry" in text
                or "pinhole improves" in text
            ),
            "OF054": lambda text: (
                "dry eye" in text
                or "dry eyes" in text
                or "blepharitis" in text
                or "gritty eyes" in text
                or "lubricants" in text
                or "artificial tears" in text
                or "lid hygiene" in text
                or "warm compresses" in text
            ),
            "OF055": lambda text: (
                "pvd" in text
                or "posterior vitreous detachment" in text
                or "vitreous detachment" in text
                or "acute pvd" in text
                or "weiss ring" in text
            ),
            "OF056": lambda text: (
                (
                    "retinal tear" in text
                    or "retinal hole" in text
                    or "retinal break" in text
                    or "horseshoe tear" in text
                    or "operculated hole" in text
                    or "lattice with hole" in text
                )
                and not self.is_negated(text, "retinal tear")
                and not self.is_negated(text, "retinal hole")
                and not self.is_negated(text, "retinal break")
            ),
            "OF057": lambda text: (
                (
                    "vitreous haemorrhage" in text
                    or "vitreous hemorrhage" in text
                    or "shafer sign" in text
                    or "tobacco dust" in text
                    or "pigment cells" in text
                    or "vitreous cells" in text
                )
                and not self.is_negated(text, "vitreous haemorrhage")
                and not self.is_negated(text, "vitreous hemorrhage")
                and not self.is_negated(text, "shafer sign")
                and not self._diabetic_context(text)
            ),
            "OF058": lambda text: (
                "previous retinal detachment" in text
                or "previous rd" in text
                or "history of retinal detachment" in text
                or "retinal laser" in text
                or ("retinal tear" in text and "laser" in text)
                or "cryotherapy" in text
                or "high myopia" in text
                or "lattice degeneration" in text
                or "aphakia" in text
            ),
            "OF059": lambda text: (
                "eye trauma" in text
                or "ocular trauma" in text
                or "blunt trauma" in text
                or "trauma with floaters" in text
                or "trauma with flashes" in text
                or "traumatic floaters" in text
            ),
            "OF060": lambda text: (
                "not triaged" in text
                or "awaiting triage" in text
                or "referral not found" in text
                or "referral id" in text
                or "referral received" in text
                or "re-refer" in text
                or "re refer" in text
                or "resend referral" in text
                or "lost in the system" in text
                or "issue with the referral" in text
                or "new referral is needed" in text
                or "make a new referral" in text
                or "single point" in text
                or "spoa" in text
                or "wyspoa" in text
            ),
            "OF061": lambda text: (
                "under hes" in text
                or "already under hes" in text
                or "under your care" in text
                or "under the eye clinic" in text
                or "next appointment" in text
                or "follow up appointment" in text
                or "follow-up appointment" in text
                or "fu appointment" in text
                or "due to be seen" in text
                or "not been seen" in text
                or "missed appointment" in text
                or "arrange another appointment" in text
                or "seen sooner" in text
                or "discharged from hes" in text
            ),
            "OF062": lambda text: (
                "waiting list" in text
                or "not on the waiting list" in text
                or "on waiting list" in text
                or "listed for surgery" in text
                or "listed for treatment" in text
                or "listed for cataract" in text
                or "listed for right eye" in text
                or "listed for left eye" in text
                or "expected wait time" in text
                or "wait situation" in text
                or "consented and listed" in text
            ),
            "OF063": lambda text: (
                "medisoft" in text
                or "upload results" in text
                or "add to the patient record" in text
                or "add these to the patient record" in text
                or "discharge letter" in text
                or "dc number" in text
                or "post op report" in text
                or "post-op report" in text
                or "hospital notes" in text
                or "clinic letter" in text
                or "clinic letters" in text
                or "latest hes rx" in text
                or "patient record" in text
                or "send the results" in text
            ),
            "OF064": lambda text: (
                "cannot use a mobile device" in text
                or "unable to access single point" in text
                or "send a letter" in text
                or "learning difficulties" in text
                or "contact the patient" in text
                or "organise travel" in text
                or "organize travel" in text
                or "travel needs" in text
                or "missed appointment" in text
                or "could not attend" in text
                or "cancelled appointment" in text
                or "cannot get through" in text
                or "answering machine" in text
                or "no contact details" in text
            ),
            "OF065": lambda text: (
                "no symptoms after cataract" in text
                or "no red eye pain photophobia" in text
                or "no signs of inflammation" in text
                or "asymptomatic post cataract" in text
                or "completely asymptomatic" in text
                or "expected finding after cataract" in text
                or "post cataract check normal" in text
                or "no active oedema" in text
                or "no corneal issues" in text
                or "no complications" in text
                or "vision good after cataract" in text
                or self._post_op_red_flags_are_negated(text)
            ),
            "OF066": lambda text: (
                "post cataract rx" in text
                or "post op sight test" in text
                or "post cataract assessment results" in text
                or "add to the patient record" in text
                or "add these to the patient record" in text
                or "medisoft access" in text
                or "upload the post op report" in text
                or "upload post op report" in text
                or "discharge letter" in text
                or "dc number" in text
                or "post op paperwork" in text
                or "send the results" in text
            ),
            "OF067": lambda text: (
                "expected outcome" in text
                or "expected after cataract" in text
                or "is this known" in text
                or "already monitored" in text
                or "review hospital notes" in text
                or "hospital notes" in text
                or "previous records" in text
                or "no previous records" in text
                or "physiological variation" in text
                or "beginning of post op complications" in text
                or "already under hes" in text
                or "under hes review" in text
            ),
            "OF068": lambda text: (
                "post op cmo" in text
                or "post cataract cmo" in text
                or "cystoid macular oedema" in text
                or "cystoid macular edema" in text
                or " cmo" in text
                or "cystic spaces" in text
                or "macular cyst" in text
                or "post operative cmo" in text
                or "subtle post op cmo" in text
            ),
            "OF069": lambda text: self._optic_disc_swelling_context(text),
            "OF070": lambda text: self._papilloedema_symptom_context(text),
            "OF071": lambda text: (
                "tilted disc" in text
                or "tilted discs" in text
                or "tilted nerve" in text
                or "crowded disc" in text
                or "small crowded disc" in text
                or "pseudopapilloedema" in text
                or "pseudopapilledema" in text
                or "optic disc drusen" in text
                or "disc drusen" in text
                or "anomalous disc" in text
                or "myelinated nerve fibres" in text
                or "myelinated nerve fibers" in text
                or "spontaneous venous pulsation" in text
                or "svp visible" in text
            ),
            "OF072": lambda text: self._neuro_ophthalmology_context(text),
            "OF073": lambda text: (
                "previous optic nerve referral" in text
                or "previously reviewed by hes" in text
                or "hes advised normal" in text
                or "advised within normal ranges" in text
                or "within normal ranges" in text
                or "planned review" in text
                or "no fresh concerns" in text
                or "no new referral indicated" in text
                or ("under hes" in text and ("optic nerve" in text or "disc" in text))
            ),
            "OF074": lambda text: self._iris_anterior_segment_context(text),
            "OF075": lambda text: self._iris_red_flag_context(text),
        }

        for feature_id, matcher in fallback_patterns.items():
            if feature_id in detected_ids or not matcher(q):
                continue
            detected.append({
                "Feature ID": feature_id,
                "Feature": self.get_entity_name(feature_id),
                "Matched Terms": ["fallback clinical phrase pattern"],
            })
            detected_ids.add(feature_id)

        return detected

    def _oct_term_is_glaucoma_context(self, normalised_text):
        return "rnfl" in normalised_text or "oct rnfl" in normalised_text

    def _visual_fields_are_missing_context(self, normalised_text):
        return (
            "visual fields not supplied" in normalised_text
            or "no visual fields supplied" in normalised_text
            or "fields not supplied" in normalised_text
            or "visual fields not available" in normalised_text
            or "random central misses" in normalised_text
            or ("no visual acuity" in normalised_text and "visual fields supplied" in normalised_text)
        )

    def _contact_lens_is_negated(self, normalised_text):
        return (
            "not a contact lens wearer" in normalised_text
            or "no contact lens" in normalised_text
            or "no contact lenses" in normalised_text
            or "does not wear contact lens" in normalised_text
            or "does not wear contact lenses" in normalised_text
        )

    def _cataract_context(self, normalised_text):
        return (
            "cataract" in normalised_text
            or "post cat" in normalised_text
            or "post op sight test" in normalised_text
            or "pseudophakia" in normalised_text
            or "pseudophakic" in normalised_text
        )

    def _post_op_red_flags_are_negated(self, normalised_text):
        negated_phrases = (
            "no red eye",
            "no pain",
            "no photophobia",
            "no symptoms",
            "no other symptoms",
            "no signs of inflammation",
            "no active inflammation",
            "no active oedema",
            "no active edema",
            "no corneal issues",
            "no complications",
            "completely asymptomatic",
            "asymptomatic",
        )
        if not any(phrase in normalised_text for phrase in negated_phrases):
            return False
        return (
            "post cataract" in normalised_text
            or "post cat" in normalised_text
            or "post op" in normalised_text
            or "cataract surgery" in normalised_text
            or "after cataract" in normalised_text
        )

    def _diabetic_context(self, normalised_text):
        return (
            "diabetes" in normalised_text
            or "diabetic" in normalised_text
            or "dmo" in normalised_text
            or "diabetic retinopathy" in normalised_text
            or "diabetic maculopathy" in normalised_text
        )

    def _watery_eye_is_lid_context(self, normalised_text):
        return (
            "ectropion" in normalised_text
            or "entropion" in normalised_text
            or "lid malposition" in normalised_text
            or "watery eye from lid" in normalised_text
        )

    def _orbital_context(self, normalised_text):
        return (
            "proptosis" in normalised_text
            or "orbital swelling" in normalised_text
            or "orbital signs" in normalised_text
            or "painful eye movements" in normalised_text
            or "diplopia with proptosis" in normalised_text
        )

    def _adult_diplopia_or_neuro_context(self, normalised_text):
        if self._orbital_context(normalised_text):
            return False

        positive_phrases = (
            "new diplopia",
            "sudden diplopia",
            "adult onset diplopia",
            "new double vision",
            "headache with diplopia",
            "ptosis with diplopia",
        )
        if any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in positive_phrases):
            return True

        if "new" in normalised_text and "double vision" in normalised_text and not self.is_negated(normalised_text, "double vision"):
            return True

        return (
            "neurological symptoms" in normalised_text
            and not self.is_negated(normalised_text, "neurological symptoms")
        )

    def _optic_disc_swelling_context(self, normalised_text):
        if "pseudopapilloedema" in normalised_text or "pseudopapilledema" in normalised_text:
            return False

        swelling_phrases = (
            "indistinct optic nerve margin",
            "indistinct optic nerve margins",
            "indistinct disc margin",
            "indistinct disc margins",
            "blurred disc margin",
            "blurred disc margins",
            "optic disc swelling",
            "disc swelling",
            "swollen disc",
            "swollen optic disc",
            "optic nerve swelling",
            "papilloedema",
            "papilledema",
            "elevated disc",
            "disc elevation",
        )
        if not any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in swelling_phrases):
            return False
        return not (
            "glaucoma suspect" in normalised_text
            and "indistinct" not in normalised_text
            and "swollen" not in normalised_text
            and "papilloedema" not in normalised_text
            and "papilledema" not in normalised_text
        )

    def _papilloedema_symptom_context(self, normalised_text):
        if self.is_negated(normalised_text, "neurological symptoms"):
            return False
        if "pseudopapilloedema" in normalised_text or "pseudopapilledema" in normalised_text:
            return False

        direct_phrases = (
            "papilloedema",
            "papilledema",
            "raised intracranial pressure",
            "raised icp",
            "transient visual obscurations",
            "pulsatile tinnitus",
            "headache with vomiting",
            "headaches with vomiting",
            "severe headache",
            "vomiting with headache",
            "sixth nerve palsy",
        )
        if any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in direct_phrases):
            return True

        optic_disc_context = self._optic_disc_swelling_context(normalised_text)
        headache_context = "headache" in normalised_text or "headaches" in normalised_text
        neuro_context = "diplopia" in normalised_text or "vomiting" in normalised_text or "neurological symptoms" in normalised_text
        return optic_disc_context and headache_context and neuro_context

    def _neuro_ophthalmology_context(self, normalised_text):
        if (
            "proptosis" in normalised_text
            or "orbital swelling" in normalised_text
            or "orbital signs" in normalised_text
            or "diplopia with proptosis" in normalised_text
        ):
            return False

        phrases = (
            "pain on eye movements",
            "painful eye movements",
            "reduced colour vision",
            "reduced color vision",
            "colour vision loss",
            "color vision loss",
            "central visual field defect",
            "central field defect",
            "rapd",
            "relative afferent pupillary defect",
            "pupil abnormality",
            "abnormal pupil",
            "optic neuritis",
        )
        if any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in phrases):
            return True

        pain_context = "pain around eye" in normalised_text or "forehead pain" in normalised_text
        visual_pathway_context = (
            "colour vision" in normalised_text
            or "color vision" in normalised_text
            or "visual field" in normalised_text
            or "pupil" in normalised_text
            or "optic nerve" in normalised_text
        )
        return pain_context and visual_pathway_context

    def _iris_anterior_segment_context(self, normalised_text):
        phrases = (
            "raised iris",
            "raised on the iris",
            "raised on iris",
            "elevated iris",
            "iris lesion",
            "iris lump",
            "iris nodule",
            "iris mass",
            "iris cyst",
            "iris naevus",
            "iris nevus",
            "pigmented iris lesion",
            "iris pigmentation",
            "iris abnormality",
        )
        if any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in phrases):
            return True
        return bool(re.search(r"\braised\b.{0,30}\biris\b", normalised_text))

    def _iris_red_flag_context(self, normalised_text):
        if not ("iris" in normalised_text or "angle" in normalised_text or "anterior chamber" in normalised_text):
            return False

        phrases = (
            "rubeosis",
            "iris neovascularisation",
            "iris neovascularization",
            "new vessels on iris",
            "hyphaema",
            "hyphema",
            "irregular pupil",
            "distorted pupil",
            "angle involvement",
            "angle closure",
            "secondary glaucoma",
            "raised iop",
            "high iop",
            "painful red eye",
        )
        return any(phrase in normalised_text and not self.is_negated(normalised_text, phrase) for phrase in phrases)

    def _field_term_is_glaucoma_context(self, normalised_text, term):
        if term not in {"field defect", "visual field defect", "field loss"}:
            return False
        glaucoma_context = (
            "glaucoma" in normalised_text
            or "optic disc" in normalised_text
            or "disc" in normalised_text
            or "iop" in normalised_text
            or "mmhg" in normalised_text
            or "visual field" in normalised_text
            or "visual fields" in normalised_text
        )
        retina_context = (
            "curtain" in normalised_text
            or "shadow" in normalised_text
            or "flashes" in normalised_text
            or "floaters" in normalised_text
        )
        return glaucoma_context and not retina_context

    def _monocular_term_is_diplopia_context(self, normalised_text):
        return "monocular diplopia" in normalised_text

    def get_entity_name(self, entity_id):
        row = self.entities[self.entities["Entity ID"] == entity_id]
        if row.empty:
            return entity_id
        return row.iloc[0]["Entity Name"]

    def get_entities_by_type(self, entity_type):
        return self.entities[self.entities["Entity Type"] == entity_type]["Entity ID"].tolist()

    def _weight(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0

    def rank_presentations(self, detected_features, top_n=5):
        detected_ids = {feature["Feature ID"] for feature in detected_features}
        presentation_ids = self.get_entities_by_type("Presentation")
        results = []

        for presentation_id in presentation_ids:
            if presentation_id == "PR050" and "OF070" not in detected_ids:
                continue
            if presentation_id == "PR053" and "OF072" not in detected_ids:
                continue

            excluded_ids = self._excluded_entities_for_presentation(presentation_id)
            if detected_ids.intersection(excluded_ids):
                continue

            rels = self.relationships[
                (self.relationships["Target Entity ID"] == presentation_id)
                & (self.relationships["Relationship Type"] == "supports")
            ]

            score = 0.0
            evidence = []
            for _, rel in rels.iterrows():
                source_id = rel["Source Entity ID"]
                weight = self._weight(rel["Weight"])
                if source_id in detected_ids:
                    score += weight
                    evidence.append({
                        "Source Entity ID": source_id,
                        "Source Entity": self.get_entity_name(source_id),
                        "Weight": weight,
                    })

            confidence = min(round(score * 100), 100)
            if confidence >= 50:
                results.append({
                    "Presentation ID": presentation_id,
                    "Presentation": self.get_entity_name(presentation_id),
                    "Confidence": confidence,
                    "Raw Score": score,
                    "Evidence": evidence,
                })

        return sorted(results, key=lambda item: (item["Confidence"], item["Raw Score"]), reverse=True)[:top_n]

    def _excluded_entities_for_presentation(self, presentation_id):
        if self.rules.empty or "Target Entity" not in self.rules.columns:
            return set()

        rules = self.rules[self.rules["Target Entity"] == presentation_id]
        excluded = set()
        for _, rule in rules.iterrows():
            value = str(rule.get("Excluded Entities", ""))
            excluded.update(part.strip() for part in value.split(";") if part.strip())
        return excluded

    def rank_safety(self, presentations, top_n=5):
        safety_scores = {}
        # V7.0 safety confidence inherits from the top-ranked presentation only.
        presentations = presentations[:1] if presentations else []

        for presentation in presentations:
            presentation_id = presentation["Presentation ID"]
            presentation_confidence = presentation["Confidence"] / 100
            rels = self.relationships[
                (self.relationships["Source Entity ID"] == presentation_id)
                & (self.relationships["Relationship Type"] == "raises concern for")
            ]

            for _, rel in rels.iterrows():
                safety_id = rel["Target Entity ID"]
                weight = self._weight(rel["Weight"])
                confidence = min(round(presentation_confidence * weight * 100), 100)
                evidence = {
                    "Presentation ID": presentation_id,
                    "Presentation": presentation["Presentation"],
                    "Presentation Confidence": presentation["Confidence"],
                    "Relationship Weight": weight,
                    "Combined Safety Confidence": confidence,
                }

                current = safety_scores.get(safety_id)
                if current is None or confidence > current["Confidence"]:
                    safety_scores[safety_id] = {
                        "Safety Condition ID": safety_id,
                        "Safety Condition": self.get_entity_name(safety_id),
                        "Confidence": confidence,
                        "Evidence": [evidence],
                    }

        return sorted(safety_scores.values(), key=lambda item: item["Confidence"], reverse=True)[:top_n]

    def missing_information(self, top_presentation):
        if not top_presentation:
            return []

        presentation_id = top_presentation["Presentation ID"]
        rels = self.relationships[
            (self.relationships["Source Entity ID"] == presentation_id)
            & (self.relationships["Relationship Type"] == "requires")
        ]

        items = []
        for _, rel in rels.iterrows():
            target_id = rel["Target Entity ID"]
            items.append({
                "Missing Information ID": target_id,
                "Missing Information": self.get_entity_name(target_id),
            })
        return items

    def recommend_outcome(self, top_presentation=None, top_safety=None, missing_info=None):
        missing_info = missing_info or []

        if top_safety and top_safety["Confidence"] >= 80:
            return self._outcome("OUT003", "High-confidence safety condition override")

        if top_presentation:
            rels = self.relationships[
                (self.relationships["Source Entity ID"] == top_presentation["Presentation ID"])
                & (self.relationships["Relationship Type"] == "recommends outcome")
            ]
            if not rels.empty:
                outcome_id = rels.iloc[0]["Target Entity ID"]
                return self._outcome(outcome_id, f"Graph recommendation from {top_presentation['Presentation ID']}")

        if missing_info:
            return self._outcome("OUT002", "Missing information present")

        return self._outcome("OUT001", "No high-risk safety condition and no graph escalation")

    def draft_response(self, result):
        outcome = result.get("Outcome Recommendation", {})
        outcome_id = outcome.get("Outcome ID", "")
        presentations = result.get("Presentation Ranking", [])
        top_presentation_id = presentations[0]["Presentation ID"] if presentations else ""
        missing_info = result.get("Missing Information", [])

        if top_presentation_id == "PR054":
            return {
                "Summary": "This looks like a raised iris / anterior segment lesion query, but the graph does not have enough information to advise safely.",
                "Suggested response": (
                    "Thanks for this. I would be happy to advise, but it would be safer to clarify a few details first. "
                    "Could you send an anterior-segment/slit-lamp photograph if available, laterality, size, location, colour/pigmentation, "
                    "whether the lesion is vascular or changing, VA, IOP, pupil shape/reaction, anterior chamber activity, angle findings, "
                    "and whether there is pain, redness, photophobia, hyphaema or rubeosis?"
                ),
                "Safety net": "If there is pain, red eye, reduced vision, high IOP, rubeosis/new iris vessels, hyphaema, pupil distortion or rapid change, please refer urgently using the local ophthalmology pathway.",
            }

        if top_presentation_id == "PR055":
            return {
                "Summary": "This query contains iris/anterior-segment red-flag features.",
                "Suggested response": (
                    "Thanks for this. The features described need urgent ophthalmology assessment. "
                    "Please refer using your local urgent anterior-segment pathway, and include VA, IOP, pupil findings, anterior chamber/angle findings and an anterior-segment image if available."
                ),
                "Safety net": "If there is reduced vision, pain, red eye, rubeosis/new iris vessels, hyphaema, pupil distortion or rapid change, this should be managed urgently rather than as routine advice.",
            }

        if top_presentation_id == "PR060":
            return {
                "Summary": "This appears to be a referral-level macular OCT structural or fluid concern.",
                "Suggested response": (
                    "Thanks for this. The OCT description suggests a macular structural/fluid concern that should be referred for medical-retina review in line with the local pathway. "
                    "It would be helpful to include OCT images, VA, laterality, symptom onset/change, distortion/Amsler status, fundus findings and any previous comparison OCT or macular history."
                ),
                "Safety net": "If there is sudden central vision loss, new distortion, rapidly worsening symptoms, haemorrhage or suspected active wet AMD/CNV, please refer urgently using the local ophthalmology pathway.",
            }

        if top_presentation_id == "PR061":
            return {
                "Summary": "This appears to be a retinal inflammatory or white-dot lesion query.",
                "Suggested response": (
                    "Thanks for this. A retinal inflammatory or white-dot lesion query is best assessed through the local retina pathway. "
                    "Please refer and include VA, laterality, onset/duration, symptoms, fundus/OCT images and any systemic or inflammatory history if available."
                ),
                "Safety net": "If there is marked or rapidly worsening vision loss, significant pain, severe inflammation or other acute red flags, please refer urgently using the local pathway.",
            }

        if top_presentation_id == "PR062":
            return {
                "Summary": "This appears to be a diabetic macular/OCT abnormality or progression concern.",
                "Suggested response": (
                    "Thanks for this. The information suggests a diabetic macular/OCT abnormality or progression concern, so referral through the local diabetic medical-retina pathway would be appropriate. "
                    "It would be helpful to include OCT/photographs, VA, laterality, retinopathy/maculopathy grade, previous laser/injections and current screening or HES status where available."
                ),
                "Safety net": "If there are new vessels, vitreous haemorrhage, sudden vision loss or rapidly progressive diabetic eye disease features, please refer urgently using the local pathway.",
            }

        if top_presentation_id == "PR063":
            return {
                "Summary": "This appears to be a post-cataract CMO, VMT or cystic macular change concern.",
                "Suggested response": (
                    "Thanks for this. The post-operative OCT findings suggest possible CMO/VMT or cystic macular change, so referral through the local post-operative cataract or medical-retina pathway would be appropriate. "
                    "It would be helpful to include OCT images, VA, time since surgery, symptom change, current drops/treatment and whether there are any post-operative infection red flags."
                ),
                "Safety net": "If there is pain, red eye, photophobia, hypopyon, marked reduced vision or concern about endophthalmitis, please refer urgently using the local post-operative pathway.",
            }

        if top_presentation_id == "PR010":
            return {
                "Summary": "This appears to be a glaucoma or optic-nerve structural/visual-field deterioration concern.",
                "Suggested response": (
                    "Thanks for this. The field/disc/OCT information suggests possible glaucoma or optic-nerve progression, so referral or review through the local glaucoma pathway would be appropriate. "
                    "It would be helpful to include IOP values and method, disc/OCT RNFL images, visual-field printouts/reliability, previous comparison and whether the patient is already under HES/glaucoma follow-up."
                ),
                "Safety net": "If there are acute angle-closure symptoms, sudden vision loss, painful red eye or rapidly progressive field/optic-nerve change, please refer urgently using the local pathway.",
            }

        if outcome_id == "OUT003":
            return {
                "Summary": "The graph has identified features that may need urgent assessment.",
                "Suggested response": (
                    "Thanks for this. From the details provided, this needs ophthalmology assessment. "
                    "Please refer using the appropriate local pathway and include symptom onset, VA, laterality, key positive and negative symptoms, relevant examination findings and any images/OCT/photos available."
                ),
                "Safety net": "If symptoms are acute, severe, rapidly worsening, or associated with pain, red eye, neurological symptoms or sudden reduced vision, please refer urgently rather than via a routine pathway.",
            }

        if outcome_id == "OUT002":
            info = ", ".join(item["Missing Information"] for item in missing_info) if missing_info else "clear clinical details, key symptoms, VA, laterality, relevant examination findings and images/OCT/photos where available"
            return {
                "Summary": "More information is needed before safe advice can be given.",
                "Suggested response": f"Thanks for this. I can advise more safely with a little more detail. Could you send {info}?",
                "Safety net": "If there are new severe symptoms, reduced vision, pain, red eye, neurological symptoms or other red flags, please refer urgently using the local pathway rather than waiting for advice.",
            }

        return {
            "Summary": "The graph did not identify a high-risk presentation from the supplied text.",
            "Suggested response": "Thanks for this. From the information provided, this sounds suitable for advice back to the referrer, as long as there are no red-flag symptoms or concerning examination findings.",
            "Safety net": "Please advise the patient to seek urgent reassessment if symptoms worsen, vision drops, pain/redness develops, or any other red flags appear.",
        }

    def _outcome(self, outcome_id, rationale):
        row = self.entities[self.entities["Entity ID"] == outcome_id]
        if not row.empty:
            name = row.iloc[0]["Entity Name"]
        elif not self.outcome_mapping.empty and "Outcome ID" in self.outcome_mapping.columns:
            match = self.outcome_mapping[self.outcome_mapping["Outcome ID"] == outcome_id]
            name = match.iloc[0]["Outcome Name"] if not match.empty else outcome_id
        else:
            name = outcome_id
        return {"Outcome ID": outcome_id, "Outcome": name, "Rationale": rationale}

    def analyse(self, query):
        detected_features = self.detect_features(query)
        presentations = self.rank_presentations(detected_features)
        top_presentation = presentations[0] if presentations else None
        safety = self.rank_safety(presentations)
        top_safety = safety[0] if safety else None
        missing_info = self.missing_information(top_presentation)
        outcome = self.recommend_outcome(top_presentation, top_safety, missing_info)
        if not top_presentation and not detected_features and query.strip():
            missing_info = [{
                "Missing Information ID": "MI000",
                "Missing Information": "Main eye problem or question, which eye is affected, vision/VA, symptom duration, key symptoms, relevant examination findings and any photo/OCT/image if available",
            }]
            outcome = self._outcome(
                "OUT002",
                "No recognised graph features in a non-empty clinical query",
            )
        if not top_presentation and detected_features and outcome["Outcome ID"] == "OUT001":
            outcome = self._outcome(
                "OUT002",
                "Recognised clinical features but no confident graph presentation",
            )
        result = {
            "Query": query,
            "Detected Features": detected_features,
            "Presentation Ranking": presentations,
            "Safety Ranking": safety,
            "Missing Information": missing_info,
            "Outcome Recommendation": outcome,
            "Audit": {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Knowledge Graph": str(self.graph_file),
                "Engine Version": "EyeV OKG V7.0 locked baseline",
            },
        }
        result["Draft Response"] = self.draft_response(result)
        return result

    def print_report(self, result):
        print("=" * 70)
        print("EyeV Clinical Reasoning Report")
        print("=" * 70)

        print("\nQUESTION")
        print("-" * 70)
        print(result["Query"])

        print("\nDETECTED FEATURES")
        print("-" * 70)
        if result["Detected Features"]:
            for feature in result["Detected Features"]:
                print(f"- {feature['Feature ID']} {feature['Feature']}")
        else:
            print("No clinical features detected.")

        print("\nPRESENTATION RANKING")
        print("-" * 70)
        if result["Presentation Ranking"]:
            for idx, presentation in enumerate(result["Presentation Ranking"], start=1):
                print(f"{idx}. {presentation['Presentation ID']} {presentation['Presentation']} - {presentation['Confidence']}%")
        else:
            print("No confident presentation identified.")

        print("\nSAFETY")
        print("-" * 70)
        if result["Safety Ranking"]:
            for safety in result["Safety Ranking"]:
                print(f"- {safety['Safety Condition ID']} {safety['Safety Condition']} - {safety['Confidence']}%")
        else:
            print("No linked safety condition identified.")

        print("\nMISSING INFORMATION")
        print("-" * 70)
        if result["Missing Information"]:
            for item in result["Missing Information"]:
                print(f"- {item['Missing Information']}")
        else:
            print("No graph-defined missing information requirements identified.")

        outcome = result["Outcome Recommendation"]
        print("\nOUTCOME")
        print("-" * 70)
        print(f"{outcome['Outcome ID']} - {outcome['Outcome']}")
        print(f"Rationale: {outcome['Rationale']}")

        print("\nAUDIT")
        print("-" * 70)
        for key, value in result["Audit"].items():
            print(f"{key}: {value}")

        print("=" * 70)





def run_locked_validation():
    engine = OKGEngine()
    query = "Patient has new flashes and floaters but no curtain or field loss."
    result = engine.analyse(query)
    engine.print_report(result)

    assert [feature["Feature ID"] for feature in result["Detected Features"]] == ["OF004", "OF005"]
    assert result["Presentation Ranking"][0]["Presentation ID"] == "PR004"
    assert result["Presentation Ranking"][0]["Confidence"] == 80
    assert result["Safety Ranking"][0]["Safety Condition ID"] == "SC002"
    assert result["Safety Ranking"][0]["Confidence"] == 36
    assert result["Outcome Recommendation"]["Outcome ID"] == "OUT002"
    print("V7.0 locked validation passed.")
    return result


if __name__ == "__main__":
    run_locked_validation()
