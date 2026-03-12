from fastapi import FastAPI
from pydantic import BaseModel
# Import the functions and models from your existing script
from predict import (
    clean_text, extract_urls, extract_url_features,
    spam_model, spam_tfidf, url_model, URL_LABEL_MAP, DANGEROUS_URL_TYPES
)

app = FastAPI(title="Spam & Malicious URL Detector API")

# Define the input format expected from the user
class PredictRequest(BaseModel):
    text: str

# Define the API endpoint
@app.post("/predict")
def predict_message(request: PredictRequest):
    text = request.text
    if not text:
        return {"error": "Text cannot be empty"}

    # 1. Spam check
    vec = spam_tfidf.transform([clean_text(text)])
    spam_label = int(spam_model.predict(vec)[0])
    spam_conf = float(spam_model.predict_proba(vec)[0][spam_label])
    spam_str = "SPAM" if spam_label == 1 else "HAM"

    # 2. URL check
    urls = extract_urls(text)
    
    # 3. Verdict logic
    has_danger_url = False
    url_results = []
    
    for url in urls:
        feat = extract_url_features(url)
        url_label = int(url_model.predict(feat)[0])
        url_conf = float(url_model.predict_proba(feat)[0][url_label])
        url_str = URL_LABEL_MAP[url_label]
        
        dangerous = url_str in DANGEROUS_URL_TYPES
        if dangerous:
            has_danger_url = True
            
        url_results.append({
            "url": url, 
            "label": url_str.upper(), 
            "confidence": url_conf, 
            "dangerous": dangerous
        })

    if spam_label == 1 and has_danger_url:
        verdict = "DANGEROUS"
    elif spam_label == 1 or has_danger_url:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"

    # 4. Return result as JSON
    return {
        "message": {
            "prediction": spam_str,
            "confidence": spam_conf
        },
        "urls": url_results,
        "verdict": verdict
    }
