"""
Synthetic payer policy corpus. These are invented for engineering purposes and do
NOT reflect any real payer's coverage criteria.
"""
from __future__ import annotations

from priorauth.schemas import Criterion, Policy

A1C = "4548-4"
EGFR = "62238-1"
BMI = "39156-5"
AHI = "SYN-AHI"      # synthetic code: apnea-hypopnea index


def C(id, text, kind, required=True, **params) -> Criterion:
    return Criterion(id=id, text=text, kind=kind, params=params, required=required)


POLICIES: list[Policy] = [
    Policy(policy_id="SYN-MRI-LSPINE", title="MRI Lumbar Spine", service_codes=["72148"],
        summary="Lumbar MRI is covered for adults with low back pain persisting after at least six weeks of conservative therapy.",
        criteria=[
            C("age", "Patient is 18 years or older", "age", min=18),
            C("dx", "Diagnosis of low back pain or lumbar disc disorder", "diagnosis", prefixes=["M54.5", "M51"]),
            C("duration", "Symptoms persisting at least 42 days", "duration", prefixes=["M54.5", "M51"], min_days=42),
            C("conservative", "NSAID trial of at least 14 days", "prior_therapy", classes=["nsaid"], min_days=14),
            C("pt", "Physical therapy or home exercise documented", "documentation", keywords=["physical therapy", "home exercise"]),
        ]),
    Policy(policy_id="SYN-GLP1-T2D", title="GLP-1 Receptor Agonist for Type 2 Diabetes", service_codes=["GLP1-SEMA", "GLP1-DULA"],
        summary="GLP-1 RA is covered for adults with type 2 diabetes inadequately controlled after at least 90 days of metformin.",
        criteria=[
            C("age", "Patient is 18 years or older", "age", min=18),
            C("dx", "Diagnosis of type 2 diabetes", "diagnosis", prefixes=["E11"]),
            C("a1c", "HbA1c of 7.0% or higher within 90 days", "lab_threshold", loinc=A1C, op=">=", value=7.0, within_days=90),
            C("metformin", "Metformin for at least 90 days", "prior_therapy", classes=["metformin"], min_days=90),
        ]),
    Policy(policy_id="SYN-CGM", title="Continuous Glucose Monitor", service_codes=["A9276", "E2103"],
        summary="CGM is covered for patients with diabetes on insulin who have had a recent in-person visit.",
        criteria=[
            C("dx", "Diagnosis of diabetes", "diagnosis", prefixes=["E10", "E11"]),
            C("insulin", "Insulin therapy for at least 30 days", "prior_therapy", classes=["insulin"], min_days=30),
            C("visit", "In-person visit documented", "documentation", keywords=["in-person visit", "office visit", "face-to-face"]),
        ]),
    Policy(policy_id="SYN-PUMP", title="Insulin Pump", service_codes=["E0784"],
        summary="Insulin pump is covered for type 1 diabetes with persistent hyperglycemia on multiple daily injections.",
        criteria=[
            C("dx", "Diagnosis of type 1 diabetes", "diagnosis", prefixes=["E10"]),
            C("a1c", "HbA1c of 7.5% or higher within 90 days", "lab_threshold", loinc=A1C, op=">=", value=7.5, within_days=90),
            C("mdi", "Multiple daily insulin injections for at least 90 days", "prior_therapy", classes=["insulin"], min_days=90),
            C("education", "Diabetes self-management education completed", "documentation", keywords=["diabetes education", "dsme"]),
        ]),
    Policy(policy_id="SYN-TKA", title="Total Knee Arthroplasty", service_codes=["27447"],
        summary="TKA is covered for advanced knee osteoarthritis after at least 90 days of conservative management.",
        criteria=[
            C("age", "Patient is 18 years or older", "age", min=18),
            C("dx", "Diagnosis of knee osteoarthritis", "diagnosis", prefixes=["M17"]),
            C("duration", "Symptoms persisting at least 90 days", "duration", prefixes=["M17"], min_days=90),
            C("imaging", "Radiographic evidence of advanced arthritis", "documentation", keywords=["kellgren", "bone-on-bone", "joint space narrowing"]),
            C("conservative", "NSAID or injection trial documented", "prior_therapy", classes=["nsaid", "corticosteroid-injection"], min_days=30),
        ]),
    Policy(policy_id="SYN-BIOLOGIC-RA", title="Biologic DMARD for Rheumatoid Arthritis", service_codes=["J0135"],
        summary="Biologic DMARD is covered for RA after an inadequate response to methotrexate.",
        criteria=[
            C("age", "Patient is 18 years or older", "age", min=18),
            C("dx", "Diagnosis of rheumatoid arthritis", "diagnosis", prefixes=["M05", "M06"]),
            C("mtx", "Methotrexate for at least 90 days", "prior_therapy", classes=["methotrexate"], min_days=90),
            C("activity", "Disease activity score documented", "documentation", keywords=["das28", "cdai"]),
        ]),
    Policy(policy_id="SYN-CPAP", title="CPAP Device", service_codes=["E0601"],
        summary="CPAP is covered for obstructive sleep apnea confirmed by sleep study with AHI of 15 or higher.",
        criteria=[
            C("dx", "Diagnosis of obstructive sleep apnea", "diagnosis", prefixes=["G47.33"]),
            C("ahi", "AHI of 15 or higher", "lab_threshold", loinc=AHI, op=">=", value=15, within_days=730),
            C("f2f", "Face-to-face evaluation documented", "documentation", keywords=["face-to-face", "in-person visit"]),
        ]),
    Policy(policy_id="SYN-SLEEP-STUDY", title="Polysomnography", service_codes=["95810"],
        summary="Sleep study is covered when sleep apnea is suspected and sleepiness is documented.",
        criteria=[
            C("dx", "Suspected sleep apnea or related symptom", "diagnosis", prefixes=["G47.3", "R06.83"]),
            C("epworth", "Epworth sleepiness scale documented", "documentation", keywords=["epworth"]),
        ]),
    Policy(policy_id="SYN-BARIATRIC", title="Bariatric Surgery", service_codes=["43644"],
        summary="Bariatric surgery is covered for BMI of 40 or higher, or 35 or higher with a major comorbidity, after a supervised diet program.",
        criteria=[
            C("age", "Patient is 18 years or older", "age", min=18),
            C("dx", "Diagnosis of obesity", "diagnosis", prefixes=["E66", "Z68.3", "Z68.4"]),
            C("bmi", "BMI of 35 or higher", "lab_threshold", loinc=BMI, op=">=", value=35, within_days=180),
            C("diet", "Supervised diet program documented", "documentation", keywords=["supervised diet", "medically supervised weight"]),
        ]),
    Policy(policy_id="SYN-SGLT2", title="SGLT2 Inhibitor", service_codes=["SGLT2-EMPA"],
        summary="SGLT2 inhibitor is covered for type 2 diabetes with adequate renal function after metformin.",
        criteria=[
            C("dx", "Diagnosis of type 2 diabetes", "diagnosis", prefixes=["E11"]),
            C("egfr", "eGFR of 20 or higher within 180 days", "lab_threshold", loinc=EGFR, op=">=", value=20, within_days=180),
            C("metformin", "Metformin for at least 60 days", "prior_therapy", classes=["metformin"], min_days=60),
        ]),
    Policy(policy_id="SYN-BRCA", title="BRCA1/2 Genetic Testing", service_codes=["81162"],
        summary="BRCA testing is covered for patients with breast cancer or a documented family history.",
        criteria=[
            C("dx", "Breast cancer or family history of breast cancer", "diagnosis", prefixes=["C50", "Z80.3"]),
            C("counseling", "Genetic counseling documented", "documentation", keywords=["genetic counseling"]),
        ]),
    Policy(policy_id="SYN-PT-EXT", title="Extended Physical Therapy", service_codes=["97110"],
        summary="Physical therapy beyond 20 visits is covered when measurable functional progress is documented.",
        criteria=[
            C("dx", "Musculoskeletal diagnosis", "diagnosis", prefixes=["M", "S"]),
            C("progress", "Measurable functional progress documented", "documentation", keywords=["functional progress", "improved range of motion", "improved strength"]),
        ]),
]

BY_ID = {p.policy_id: p for p in POLICIES}
BY_CODE = {c: p for p in POLICIES for c in p.service_codes}
