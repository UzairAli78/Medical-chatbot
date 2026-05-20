from flask import Flask, render_template, jsonify, request
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
from dotenv import load_dotenv
import joblib
import markdown
import re
import unicodedata
import os
from PIL import Image
import io

from src.helper import download_hugging_face_embeddings
from src.prompt import *

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ── Load embeddings & Pinecone ─────────────────────────────────────────────────
embeddings   = joblib.load("embeddings.joblib")
index_name   = "medical-chatbot"
pc           = Pinecone(api_key=PINECONE_API_KEY)
index        = pc.Index(index_name)
bm25_encoder = BM25Encoder().load("bm25_values.joblib")

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash",
    api_version="v1",
    name="gemini",
    temperature=0.7
)

# RAG pipeline — for medical questions
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Direct Gemini model — for casual conversation
casual_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=(
        "You are MediChat AI, a friendly and warm medical assistant chatbot. "
        "When users greet you or make small talk, respond warmly and naturally, "
        "then gently invite them to ask a medical question. "
        "Keep responses concise and friendly. You can chat casually but your "
        "main purpose is helping with medical questions."
    )
)

# ── Input sanitizer ────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    """
    Remove emojis and non-ASCII characters that break the BM25 encoder.
    Keeps standard Latin characters, digits, punctuation, and spaces.
    """
    # Normalize unicode first
    text = unicodedata.normalize("NFKD", text)
    # Remove anything that isn't basic ASCII printable text
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Collapse extra whitespace
    text = " ".join(text.split())
    return text.strip()

# ── Smart router ───────────────────────────────────────────────────────────────
MEDICAL_KEYWORDS = [
    "symptom", "symptoms", "disease", "fever", "pain", "headache", "doctor",
    "medicine", "medication", "drug", "treatment", "diagnosis", "cause",
    "cure", "infection", "virus", "bacteria", "blood", "heart", "lung",
    "kidney", "liver", "cancer", "diabetes", "allergy", "allergic", "dose",
    "dosage", "prescribe", "prescription", "surgery", "hospital", "health",
    "throat", "cough", "cold", "flu", "vomit", "nausea", "diarrhea",
    "rash", "swelling", "injury", "wound", "bone", "muscle", "nerve",
    "stomach", "chest", "breathing", "dizzy", "dizziness", "fatigue",
    "tired", "sleep", "mental", "anxiety", "depression", "weight", "diet",
    "nutrition", "vitamin", "supplement", "vaccine", "immunity", "chronic",
    "acute", "condition", "disorder", "syndrome", "medical", "clinical",
    "patient", "physician", "specialist", "side effect", "what causes",
    "ache", "sore", "inflammation", "i have", "i feel", "i am experiencing",
    "what is", "how to treat", "should i take", "can i take", "is it safe",
]

CASUAL_PHRASES = [
    "hello", "hi", "hey", "hii", "helo", "good morning", "good afternoon",
    "good evening", "how are you", "what's up", "whats up", "sup",
    "who are you", "what can you do", "tell me about yourself",
    "what are you", "nice to meet", "thanks", "thank you", "ok", "okay",
    "cool", "great", "awesome", "bye", "goodbye", "see you", "later",
]

def is_casual(message: str) -> bool:
    msg = message.lower().strip()
    # If any medical keyword present → always route to RAG
    if any(kw in msg for kw in MEDICAL_KEYWORDS):
        return False
    # Matches a casual greeting phrase → casual
    for phrase in CASUAL_PHRASES:
        if (msg == phrase
                or msg.startswith(phrase + " ")
                or msg.startswith(phrase + "!")
                or msg.startswith(phrase + ",")):
            return True
    # Short message with no medical keywords → treat as casual
    if len(msg.split()) <= 4:
        return True
    return False

# ── Chat route ─────────────────────────────────────────────────────────────────
@app.route("/get", methods=["GET", "POST"])
def chat():
    raw_msg = request.form["msg"]
    clean_msg = sanitize(raw_msg)   # ← FIX: strip emojis before any processing
    print(f"User (raw): {raw_msg}")
    print(f"User (clean): {clean_msg}")

    if not clean_msg:
        return "Hey! Could you rephrase that? I didn't quite catch it 😊"

    if is_casual(clean_msg):
        # Casual conversation — direct Gemini, no book search needed
        try:
            response = casual_model.generate_content(clean_msg)
            response_text = response.text
        except Exception as e:
            print(f"Casual model error: {e}")
            response_text = (
                "Hey there! I'm MediChat AI, your medical assistant. "
                "Feel free to ask me any medical question and I'll do my best to help!"
            )
    else:
        # Medical question — run through RAG (book search + Gemini)
        try:
            result = qa.invoke({"query": clean_msg})
            response_text = result.get("result", "")
            if not response_text.strip():
                response_text = (
                    "I couldn't find specific information about that in my knowledge base. "
                    "Please consult a qualified doctor for accurate advice."
                )
        except Exception as e:
            print(f"RAG error: {e}")
            response_text = (
                "I'm sorry, I had trouble processing that. "
                "Could you rephrase your question?"
            )

    print(f"Response: {response_text}")
    return markdown.markdown(response_text)


# ── Calorie estimator ──────────────────────────────────────────────────────────
def get_gemini_response(image_blob, prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content([prompt, image_blob])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.read()
        Image.open(io.BytesIO(bytes_data))   # validate image
        return {"mime_type": uploaded_file.content_type, "data": bytes_data}
    raise FileNotFoundError("No file uploaded")

def calculate_total_calories(response_text):
    calorie_values = re.findall(r"(\d+)\s*calories", response_text)
    return sum(map(int, calorie_values))

@app.route("/calculate_calories", methods=["POST"])
def calculate_calories():
    uploaded_file = request.files.get("image")
    if uploaded_file:
        image_data = input_image_setup(uploaded_file)
        input_prompt = """
        You are an expert nutritionist. Identify each food item visible in the image
        and estimate its calorie content per serving. Use this format:

        1. **Food Item**: Number of calories per serving
        2. **Food Item**: Number of calories per serving

        Provide the total calories at the end.
        """
        response_text     = get_gemini_response(image_data, input_prompt)
        total_calories    = calculate_total_calories(response_text)
        response_html     = markdown.markdown(response_text)
        formatted         = (
            f"<h3>Calorie Breakdown:</h3>{response_html}"
            f"<p><strong>Total Calories: </strong>{total_calories}</p>"
        )
        return jsonify({"response": formatted})
    return jsonify({"error": "No image uploaded"}), 400


# ── Page routes ────────────────────────────────────────────────────────────────
@app.route("/")
def agnos():    return render_template("index.html")

@app.route("/nutra")
def nutra():    return render_template("nutra.html")

@app.route("/chatbox")
def chatbox():  return render_template("chatbox.html")

@app.route("/login")
def login():    return render_template("Login.html")

@app.route("/signup")
def signup():   return render_template("signUp.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)