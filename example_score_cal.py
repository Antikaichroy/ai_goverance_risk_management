from typing import List, Dict
from gliner import GLiNER
import numpy as np
from sample_sentences import pii_sentences, non_pii_sentences

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


def calculate_hallucination_score(context_accuracy:float =  None, faithfulness:float = None, answer_relevance:float = None, alpha_ca:float = None, alpha_fa: float = None, alpha_ar: float = None, alpha_cafa: float = None) -> float: 
    """
    context_accuracy: the score for context_accuracy
    alpha_ca: the weight for context_accuracy
    faithfulness: the score for faithfulness 
    alpha_fa: the weight for faithfulness
    answer_relevance: the score for answer_relevance
    alpha_ar: weight for answer relevance

    """
    hal_chance_ca = 1 - context_accuracy
    hal_chance_fa = 1 - faithfulness
    hal_chance_ar = 1 - answer_relevance

    hallucination_score = alpha_ca*hal_chance_ca + alpha_fa*hal_chance_fa + alpha_ar*hal_chance_ar + alpha_cafa*hal_chance_ca*hal_chance_fa
    return round(hallucination_score, 2)

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

kwrgs = dict(context_accuracy=0.6, faithfulness=0.2, answer_relevance=0.4, alpha_ca = 0.4, alpha_fa = 0.4, alpha_ar = 0.1, alpha_cafa = 0.1) # for hallucination scoring

print(calculate_risk_score(risk_class_val(res), 0.6))