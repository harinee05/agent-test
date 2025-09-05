from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

SERVICE_LIST_URL = "https://selfservicelp.las-cruces.org/prod/tyler311/public/v2/services.xml?env=live"

app = FastAPI()

class MatchRequest(BaseModel):
    user_desc: str
    n_best: int = 3

class MatchResult(BaseModel):
    code: str
    name: str
    description: str
    score: float

model = SentenceTransformer('all-MiniLM-L6-v2')
services = []
embs = None

def fetch_services_live():
    j = requests.get(SERVICE_LIST_URL).json()
    svclist = j.get("services", {}).get("service", [])
    return [
        {
            'code': svc.get('service_code', ''),
            'name': svc.get('service_name', ''),
            'description': svc.get('description', '')
        }
        for svc in svclist
    ]

def build_service_embeddings(services, model):
    texts = [f"{svc['name']}: {svc['description']}" for svc in services]
    embs = model.encode(texts)
    return embs

@app.on_event("startup")
def load_all():
    global services, embs
    services = fetch_services_live()
    embs = build_service_embeddings(services, model)

@app.post("/match", response_model=List[MatchResult])
def match_service(req: MatchRequest):
    user_emb = model.encode([req.user_desc])[0]
    sims = np.inner(embs, user_emb)
    top_idx = np.argsort(-sims)
    results = []
    for rank in range(req.n_best):
        i = top_idx[rank]
        score = float(sims[i])
        svc = services[i]
        results.append(MatchResult(
            code=svc["code"], name=svc["name"], description=svc["description"], score=score
        ))
    return results
