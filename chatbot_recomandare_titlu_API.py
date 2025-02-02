from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Inițializare API
app = FastAPI(title="Book Recommendation API")

# Încărcăm modelele și datele
loaded_kmeans = joblib.load("carti_cluster_kmeans.pkl")
loaded_count = joblib.load("encoder_count_vectorizer.pkl")
loaded_count_matrix = joblib.load("count_matrix.pkl")
dataset_carti = pd.read_csv('dataset_carti_content_filtering.csv')

# Creăm matricea de similaritate
cosine_sim = cosine_similarity(loaded_count_matrix, loaded_count_matrix)

# Creăm o serie pentru a indexa cărțile după titlu
indices = pd.Series(dataset_carti['titluCarte'])

# Model pentru request
class BookRequest(BaseModel):
    title: str

# Funcția de recomandare
def recommend(title: str):
    if title not in indices.values:
        raise HTTPException(status_code=404, detail="Book not found in dataset")
    
    recommended_books = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indices = list(score_series.iloc[1:11].index)

    for i in top_10_indices:
        recommended_books.append(list(dataset_carti['titluCarte'])[i])
    
    return recommended_books

# Endpoint pentru recomandări
@app.post("/recommend")
def get_recommendations(request: BookRequest):
    recommendations = recommend(request.title)
    return {"recommended_books": recommendations}