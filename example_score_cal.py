from typing import List, Dict
from gliner import GLiNER
import numpy as np
from sample_sentences import pii_sentences, non_pii_sentences
import random


pii = pii_sentences
no_pii = non_pii_sentences

imp2class = {1.0: ["credit card number", "personal location", "address", "phone", "bank", "license number"], 0.5: ["height", "weight", "numerical salary amount", "job role"], 0.0: ["person name"]}
extractor = GLiNER.from_pretrained("nvidia/gliner-PII", map_location = "cuda") # NER Model
labels = ["person name", "address", "license number", "bank", "phone", "numerical salary amount", "personal location", "job role", "height", "weight","credit card number"]  # PII classes



def calculate_risk_score(risk_class_values = List[float], alpha = float) -> float:

    """The function calculates the risk score based on the given risk class values.
    risk_class_values: List of risk class values (e.g., [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    alpha: Weighting factor for the maximum risk class value (e.g., 0.6)"""

    if len(risk_class_values) > 0:
        max_risk = max(risk_class_values)
        mean_risk = sum(risk_class_values) / len(risk_class_values)
        N = len(risk_class_values)

        exposure = mean_risk*np.log(1 + N)

        risk_score = min(1.0, alpha*max_risk + (1-alpha)*exposure) # The risk score is capped at 1
        
    else:
        return 0.0

    return round(risk_score,2)


def calculate_hallucination_score(CA: float, FA: float, AR: float,
                                   alpha: float = 0.4,
                                   beta: float = 0.4,
                                   gamma: float = 0.2) -> float:
    
    """
    Useful for the calculation of the Hallucination Score. It is a composite measure of the AR, FA, CA
    CA: Context Accuracy Score [0,1]
    FA: Faithfulness Score [0,1]
    AR: Answer Relevance Score [0,1]

    alpha: Sensitivity for CA
    beta: Sensitivity for FA
    gamma: Sensitivity for AR
    
    alpha + beta + gamma = 1 (Must)

    """
    
    assert round(alpha + beta + gamma, 6) == 1.0, "Weights must sum to 1"
    assert all(0 <= v <= 1 for v in [CA, FA, AR]), "Scores must be in [0, 1]"
    
    trust = (CA ** alpha) * (FA ** beta) * (AR ** gamma)
    return round(1 - trust, 2)

res = extractor.predict_entities(pii[3], labels = labels)

def risk_class_val(res:List[Dict] = None) -> List[float]:
    seen = {}
    if len(res) > 0:
        target_pii = [x['label'] for x in res if x['score']>=0.7]
        for c in target_pii:
            if c in seen:
                seen[c]*=2
                continue  # class already seen

            for k, v in imp2class.items():
                if c in v:
                    seen[c] = k
                    break # found the class no need to look further
    else:
        return []
    
    return list(seen.values())

kwargs = dict(CA=0.6, FA=0.4, AR = 0.4) # for hallucination scoring

print(calculate_risk_score(risk_class_val(res), 0.6))


# sample hallucination testing

sample_ar = [x/100 for x in random.choices(range(1,100), k=100)]
sample_fa = [x/100 for x in random.choices(range(1,100), k=100)]
sample_ca = [x/100 for x in random.choices(range(1,100), k=100)]
results = []

for CA, FA, AR in zip(sample_ca, sample_fa, sample_ar):
    hallucination_score = calculate_hallucination_score(CA = CA, FA = FA, AR = AR)
    combination = {"context_accuracy": CA, "faithfulness": FA, "answer_relevance": AR}
    label = "hallucination" if hallucination_score>=0.5 else ("moderate hallucination" if hallucination_score>=0.25 else "low hallucination")
    results.append({"hallucination score":hallucination_score, "metrics": combination, "label": label})

print(results)