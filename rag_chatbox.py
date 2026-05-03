import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
with open("C:/Users/Shaelesh/Desktop/MLops/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# STEP 2: PREPARE TEXT
# -------------------------------
texts = []
titles = []

for item in data:
    combined = item["title"] + ". " + item["text_content"]
    texts.append(combined)
    titles.append(item["title"])

# -------------------------------
# STEP 3: LOAD MODEL
# -------------------------------
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings...")
embeddings = model.encode(texts)

# -------------------------------
# STEP 4: HELPER FUNCTIONS
# -------------------------------

def clean_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s) > 40]


def get_best_sentences(text, query):
    sentences = clean_sentences(text)
    
    if not sentences:
        return text[:200]
    
    sent_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]
    
    scores = cosine_similarity([query_embedding], sent_embeddings)[0]
    top_idx = scores.argsort()[-2:][::-1]
    
    return " ".join([sentences[i] for i in top_idx])


def detect_question_type(query):
    query = query.lower()
    
    if "why" in query:
        return "reason"
    elif "how" in query:
        return "process"
    elif "where" in query:
        return "location"
    elif "who" in query:
        return "person"
    else:
        return "general"


def generate_answer(query, top_indices, scores):
    q_type = detect_question_type(query)
    
    intro_map = {
        "reason": "The reason behind this is:",
        "process": "This happens in the following way:",
        "location": "This takes place in:",
        "person": "The key characters involved are:",
        "general": "Here’s what happens:"
    }
    
    answer = f"\n🧠 Answer:\n\n{intro_map[q_type]}\n\n"
    
    connectors = [
        "Firstly,",
        "Additionally,",
        "Meanwhile,",
        "As a result,"
    ]
    
    for idx, i in enumerate(top_indices):
        snippet = get_best_sentences(texts[i], query)
        connector = connectors[idx % len(connectors)]
        
        answer += f"{connector} {snippet}\n\n"
    
    answer += "👉 Overall, these events together explain the answer to your question."
    
    return answer

# -------------------------------
# STEP 5: CHAT LOOP
# -------------------------------
print("\n🤖 Advanced RAG Chatbot Ready! Type 'exit' to stop.\n")

while True:
    query = input("Ask: ")
    
    if query.lower() == "exit":
        break
    
    # ---------------------------
    # STEP 6: QUERY EMBEDDING
    # ---------------------------
    query_embedding = model.encode([query])[0]
    
    # ---------------------------
    # STEP 7: SIMILARITY SEARCH
    # ---------------------------
    scores = cosine_similarity([query_embedding], embeddings)[0]
    
    top_indices = scores.argsort()[-3:][::-1]
    
    # ---------------------------
    # STEP 8: GENERATE ANSWER
    # ---------------------------
    answer = generate_answer(query, top_indices, scores)
    
    print(answer)