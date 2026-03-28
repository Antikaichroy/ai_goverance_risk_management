# List 1: Sentences with specific PII (High-risk identifiers)
pii_sentences = [
    "Sourav lives in Kolkata he earns around 3242442, his phone is 32442323131",
    "John Doe (DL: 99-234-112) lives at 123 Maple St, Boston and works as a Lead Architect.",
    "The client uses account 0098234123 at Chase Bank and has a credit card ending in 4421.",
    "Patient weighs 82kg and is 185cm tall; contact her at 555-0102 regarding her file.",
    "Sarah Jenkins, a Surgeon, earns 350000 annually and lives at 42 High Ridge Road.",
    "Transfer the salary of 5000 to bank routing 021000021 for the Senior Consultant."
]

# List 2: Sentences without PII (Anonymized or General)
non_pii_sentences = [
    "The individual lives in a major city and earns a high annual salary.",
    "A valid driver's license and proof of residency are required for this application.",
    "Please provide your banking details through the secure portal only.",
    "The average height and weight for this demographic have remained stable this year.",
    "Professionals in the medical field often have varying shift schedules.",
    "All credit card data is encrypted at rest and never stored in plain text."
]