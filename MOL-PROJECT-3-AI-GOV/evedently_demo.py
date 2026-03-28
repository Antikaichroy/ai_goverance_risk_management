import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sentence_transformers import SentenceTransformer
import fitz
from langchain_ollama import ChatOllama
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import torch
from gliner import GLiNER
from gliner2 import GLiNER2
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently import Report
from evidently.presets import TextEvals
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from evidently.llm.options import OllamaOptions
from evidently import Dataset, DataDefinition
from evidently.descriptors import HuggingFace
from evidently.descriptors import LLMEval, ContextQualityLLMEval, PIILLMEval, ToxicityLLMEval
from langchain_ollama import ChatOllama
from datetime import datetime
import time
from hatesonar import Sonar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Use a pipeline as a high-level helper
from transformers import pipeline

pii_pipe = pipeline("text-classification", model="Roblox/roblox-pii-classifier")

# loading the embedding model 
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
semantic_chunk_vector_store = Chroma(persist_directory = "/home/antik/mol_ai_gov/MOL-PROJECT-3-AI-GOV/chroma", embedding_function = embed_model)
# contradiction checker prompt

contradiction_check = BinaryClassificationPromptTemplate(
    criteria = """Label an ANSWER as **contradictory** only if it directly contradicts any part of the REFERENCE.
    Differences in length or wording are acceptable. It is also acceptable if the ANSWER adds new details or omits information, as long as **no factual content contradicts** the REFERENCE.
    Your task is to compare factual consistency only — not completeness, relevance, or style.
    
    REFERENCE:
    =====
    {reference}
    =====
    """,
    target_category = "contradictory",
    non_target_category = "non-contradictory",
    uncertainty = "unknown",
    include_reasoning = True,
    pre_messages = [("system", "You are an expert evaluator. You will be given an ANSWER and REFERENCE.")]
)

#extractor = GLiNER.from_pretrained("nvidia/gliner-PII", map_location = "cpu")
#labels = ["person", "address", "license", "bank", "phone", "salary"]
'''labels = ['dob', 'eyecolor', 'ip', 'age', 'creditcardcvv', 'currency','pin', 'middlename',
       'creditcardissuer', 'jobtype', 'jobarea', 'ssn',
       'currencyname', 'iban', 'username', 'companyname',
       'buildingnumber', 'zipcode',
       'creditcardnumber', 'accountname',
       'gender','firstname', 'accountnumber', 'secondaryaddress',
       'email', 'phonenumber', 'lastname', 'currencycode', 'state', 'county', 'employee_id',
       'job_description', 'card_expiry_dates', 'salary',
       'personal location']
'''
def get_report(eval_df):
    
    testing_dataset = Dataset.from_pandas(
        eval_df.iloc[[-1]],
        data_definition=DataDefinition(),
        descriptors=[
            #ContextRelevance("question", "context", output_scores=True, aggregation_method="hit", method="llm"),
            ContextQualityLLMEval(
                "reference", 
                alias = "good context", 
                include_reasoning =  False,
                question = "user question", 
                provider = "ollama", 
                model = "gemini-3-flash-preview:latest"),
            LLMEval(
                "model response",
                template=contradiction_check,
                additional_columns={"reference": "reference"},
                provider="ollama",
                model="gemini-3-flash-preview:latest",
                alias="Contradictions"
            )],
        options=OllamaOptions(api_url="http://127.0.0.1:11434"))

    #report = Report([TextEvals()])
    #my_eval = report.run(testing_dataset, None)
    df = testing_dataset.as_dataframe()
    # 0)
    return {
        "good_context": df.iloc[0]["good context"],
        "contradictory": df.iloc[0]["Contradictions"]
    }


    # setting up the ollama model for the response

generator = ChatOllama(model = "gemini-3-flash-preview:latest", num_ctx = 5000)

sonar = Sonar()

def get_response(user_question:str)->str:
    """The function is going to take in the user question and return the response from the LLM"""
    start = time.perf_counter()
    all_contexts = semantic_chunk_vector_store.similarity_search(user_question, k = 10)
    final_contexts = [x.page_content for x in all_contexts]
    #print(final_contexts)
    all_contexts = semantic_chunk_vector_store.similarity_search(user_question, k = 10)
    prompt = f"""Given the context {final_contexts}, you need to answer the user question which is: user_question: {user_question} Answer in a helpful Assistant way you are the MOL AI Assistant. Do not mention from whrere you got the answer, document name, section name, avoid PII etc. Just answer as if you gave it and you knew it."""
    response = generator.invoke(prompt)
    model_response = response.content
    end = time.perf_counter()

    model_used = response.response_metadata['model_name']
    input_tokens_used = response.usage_metadata['input_tokens']
    output_tokens_used = response.usage_metadata['output_tokens']
    total_tokens_used = response.usage_metadata['total_tokens']

    time_taken = round(end-start, 2)
    current_time = datetime.now().isoformat()
    return model_response, final_contexts, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used

def upload_report(model_response, context, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used):
    df = pd.read_excel("/mnt/e/OneDrive/Documents/MOL-PROJECT-3-LOGS/logs.xlsx")
    
    #entities = extractor.predict_entities(model_response, labels)
    #print(context)
    #extracted = entities["entities"]
    max_token_size = 512
    safe_limit_char = 512*4  # 2024 lets split it with 1500 char range 
    current_pointer = 0
    next_pointer = 1500
    PII = []
    while current_pointer <= len(model_response):
        PII.extend(pii_pipe(model_response[current_pointer:next_pointer]))
        current_pointer = next_pointer 
        next_pointer += 1500
    
    score = sum([entries['score'] for entries in PII])/len(PII)

    #entities_user = pii_pipe(user_question)[0]
    #entities_resp = pii_pipe(model_response[0:1000])[0]

    #name_count = address_count = license_count = credit_count = salary_count = phone_count = 0
   
    '''for k,v in extracted.items():
        if len(v) > 0:
            PII.append(v)
        if k == "name":
            name_count = len(v)
        elif k == "address":
            address_count = len(v)
        elif k == "license":
            license_count = len(v)
        elif k == "credit card":
            credit_count = len(v)
        elif k == "salary":
            salary_count = len(v)
        elif k == "phone":
            phone_count = len(v)'''
    current_len = len(df)
    print("NO Prob")
    if score > 0.5:
        print("Prob")
        df.at[current_len, "PII"] = "DETECTED"
        #for i in entities:
            #df.at[current_len, i['label']] = 1
    elif score <= 0.5:
        print("Prob inn no PII")
        df.at[current_len, "PII"] = "NOT DETECTED"

    #checking if the response contains hatespeech

    hs = sonar.ping(text = model_response)
    classes_detected = hs['classes']
    for entries in classes_detected: # each entry is a dict
        if entries['class_name'] == "hate_speech":
            hate_speech_confidence = entries['confidence']

        elif entries['class_name'] == "offensive_language":
            offensive_language_confidence = entries['confidence']
        elif entries['class_name'] == "neither":
            neither_offensive_nor_hate_confidence = entries['confidence']

    is_hate = (hate_speech_confidence > 0.5 or offensive_language_confidence > 0.5)

    risk_level = "High" if len(PII) > 0 or is_hate else "Low"
    # storing the log containing the user question, context, model, response -->  this can be compared later with the actual ground truth
    df.at[current_len, "risk level"] = risk_level
    df.at[current_len, "user question"] = user_question
    df.at[current_len, "model response"] = model_response
    df.at[current_len, "reference"] = " ".join(context)
    #df.at[current_len, "name count"] = name_count
    #df.at[current_len, "address count"] = address_count
    #df.at[current_len, "salary"] = salary_count
    #df.at[current_len, "phone"] = phone_count
    #df.at[current_len, "license count"] = license_count
    #df.at[current_len, "credit card count"] = credit_count
    df.at[current_len, "timestamp"] = current_time
    df.at[current_len, "response time"] = time_taken 
    df.at[current_len, "model name"] = model_used
    df.at[current_len, "input tokens"] = input_tokens_used
    df.at[current_len, "output tokens"] = output_tokens_used
    df.at[current_len, "total tokens"] = total_tokens_used
    df.at[current_len, "Is Hate"] = is_hate
    df.at[current_len, "hate speech confidence"] = hate_speech_confidence
    df.at[current_len, "offensive language confidence"] = offensive_language_confidence
    df.at[current_len, "neither offensive nor hate confidence"] = neither_offensive_nor_hate_confidence
    eval_results = get_report(df)
    df.at[current_len, "good context"] = eval_results["good_context"]
    df.at[current_len, "Contradictions"] = eval_results["contradictory"]
    df.to_excel("/mnt/e/OneDrive/Documents/MOL-PROJECT-3-LOGS/logs.xlsx", index = False)
    return None

model_response, final_contexts, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used = get_response("Hi what is up")
print(model_response)
upload_report(model_response, final_contexts, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used)
    
