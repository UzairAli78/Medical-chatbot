from flask import Flask, render_template, jsonify, request
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
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
import requests
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

# Direct Gemini — for casual conversation
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


# ════════════════════════════════════════════════════════════════════════════
# OPENFDA INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

# Common drug names the system will recognise in a user's message.
# Add more entries here anytime you want to expand coverage.
KNOWN_DRUGS = [
    "ibuprofen", "paracetamol", "acetaminophen", "tylenol", "aspirin",
    "amoxicillin", "amoxil", "azithromycin", "zithromax", "doxycycline",
    "ciprofloxacin", "metformin", "glucophage", "omeprazole", "prilosec",
    "cetirizine", "zyrtec", "loratadine", "claritin", "diphenhydramine",
    "benadryl", "metoprolol", "lopressor", "lisinopril", "zestril",
    "atorvastatin", "lipitor", "losartan", "cozaar", "prednisone",
    "prednisolone", "metronidazole", "flagyl", "pantoprazole", "protonix",
    "ranitidine", "zantac", "salbutamol", "albuterol", "ventolin",
    "amlodipine", "norvasc", "clopidogrel", "plavix", "warfarin", "coumadin",
]

def extract_drug_name(message: str) -> str | None:
    """Return the first drug name found in the user message, or None."""
    msg = message.lower()
    for drug in KNOWN_DRUGS:
        if re.search(rf"\b{re.escape(drug)}\b", msg):
            return drug
    return None


def query_openfda(drug_name: str) -> dict:
    """
    Call the OpenFDA drug label endpoint and return a structured dict
    with the most useful fields.  Returns an empty dict on any failure.
    No API key required.
    """
    url = "https://api.fda.gov/drug/label.json"
    # Try brand name first, then generic name
    for field in ("openfda.brand_name", "openfda.generic_name"):
        try:
            params = {"search": f'{field}:"{drug_name}"', "limit": 1}
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code != 200:
                continue
            results = resp.json().get("results", [])
            if not results:
                continue
            r = results[0]

            def _get(key):
                """Safely grab first element of an FDA list field."""
                val = r.get(key, [""])[0]
                # FDA fields can be very long — cap at 500 chars each
                return val[:500].strip() if val else ""

            openfda = r.get("openfda", {})
            return {
                "brand_names"  : ", ".join(openfda.get("brand_name", [])[:3]),
                "generic_names": ", ".join(openfda.get("generic_name", [])[:3]),
                "purpose"      : _get("purpose") or _get("indications_and_usage"),
                "dosage"       : _get("dosage_and_administration"),
                "warnings"     : _get("warnings") or _get("warnings_and_cautions"),
                "side_effects" : _get("adverse_reactions"),
                "interactions" : _get("drug_interactions"),
                "manufacturer" : ", ".join(openfda.get("manufacturer_name", [])[:2]),
            }
        except Exception as e:
            print(f"OpenFDA query error for '{drug_name}': {e}")
            continue
    return {}


def format_fda_context(drug_name: str, fda: dict) -> str:
    """
    Turn the OpenFDA dict into a plain-text block that gets appended
    to the RAG context so Gemini can use it in its answer.
    """
    if not fda:
        return ""
    lines = [f"[OpenFDA data for {drug_name.title()}]"]
    if fda.get("brand_names"):
        lines.append(f"Brand names: {fda['brand_names']}")
    if fda.get("generic_names"):
        lines.append(f"Generic names: {fda['generic_names']}")
    if fda.get("purpose"):
        lines.append(f"Purpose / Uses: {fda['purpose']}")
    if fda.get("dosage"):
        lines.append(f"Dosage: {fda['dosage']}")
    if fda.get("warnings"):
        lines.append(f"Warnings: {fda['warnings']}")
    if fda.get("side_effects"):
        lines.append(f"Side effects: {fda['side_effects']}")
    if fda.get("interactions"):
        lines.append(f"Drug interactions: {fda['interactions']}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def sanitize(text: str) -> str:
    """Strip emojis and non-ASCII chars that break the BM25 encoder."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return " ".join(text.split()).strip()


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
] + KNOWN_DRUGS   # drug names are always medical


CASUAL_PHRASES = [
    "hello", "hi", "hey", "hii", "helo", "good morning", "good afternoon",
    "good evening", "how are you", "what's up", "whats up", "sup",
    "who are you", "what can you do", "tell me about yourself",
    "what are you", "nice to meet", "thanks", "thank you", "ok", "okay",
    "cool", "great", "awesome", "bye", "goodbye", "see you", "later",
]


def is_casual(message: str) -> bool:
    msg = message.lower().strip()
    if any(kw in msg for kw in MEDICAL_KEYWORDS):
        return False
    for phrase in CASUAL_PHRASES:
        if (msg == phrase
                or msg.startswith(phrase + " ")
                or msg.startswith(phrase + "!")
                or msg.startswith(phrase + ",")):
            return True
    if len(msg.split()) <= 4:
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# CHAT ROUTE
# ════════════════════════════════════════════════════════════════════════════

@app.route("/get", methods=["GET", "POST"])
def chat():
    raw_msg   = request.form["msg"]
    clean_msg = sanitize(raw_msg)
    print(f"User (raw): {raw_msg}")
    print(f"User (clean): {clean_msg}")

    if not clean_msg:
        return "Hey! Could you rephrase that? I didn't quite catch it."

    # ── Casual conversation ──────────────────────────────────────────────────
    if is_casual(clean_msg):
        try:
            response_text = casual_model.generate_content(clean_msg).text
        except Exception as e:
            print(f"Casual model error: {e}")
            response_text = (
                "Hey there! I'm MediChat AI, your medical assistant. "
                "Feel free to ask me any medical question!"
            )
        return markdown.markdown(response_text)

    # ── Medical question ─────────────────────────────────────────────────────
    # Step 1: Check if a specific drug is mentioned → hit OpenFDA
    drug      = extract_drug_name(clean_msg)
    fda_data  = {}
    fda_block = ""

    if drug:
        print(f"Drug detected: {drug} — querying OpenFDA...")
        fda_data  = query_openfda(drug)
        fda_block = format_fda_context(drug, fda_data)
        if fda_block:
            print("OpenFDA data retrieved successfully.")
        else:
            print("OpenFDA returned no data for this drug.")

    # Step 2: Build the query — append FDA context if we have it
    enriched_query = clean_msg
    if fda_block:
        enriched_query = (
            f"{clean_msg}\n\n"
            f"Use the following real-time FDA drug information in your answer "
            f"in addition to the knowledge base context:\n{fda_block}"
        )

    # Step 3: Run through RAG pipeline
    try:
        result        = qa.invoke({"query": enriched_query})
        response_text = result.get("result", "").strip()
        if not response_text:
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

    # Step 4: If FDA data was found, append a small source note
    if fda_data:
        response_text += (
            "\n\n---\n*Drug information sourced from the US FDA OpenFDA database.*"
        )

    print(f"Response: {response_text[:200]}...")
    return markdown.markdown(response_text)


# ════════════════════════════════════════════════════════════════════════════
# CALORIE ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

def get_gemini_response(image_blob, prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    return model.generate_content([prompt, image_blob]).text


def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.read()
        Image.open(io.BytesIO(bytes_data))          # validate
        return {"mime_type": uploaded_file.content_type, "data": bytes_data}
    raise FileNotFoundError("No file uploaded")


def calculate_total_calories(response_text):
    return sum(map(int, re.findall(r"(\d+)\s*calories", response_text)))


@app.route("/calculate_calories", methods=["POST"])
def calculate_calories():
    uploaded_file = request.files.get("image")
    if uploaded_file:
        image_data    = input_image_setup(uploaded_file)
        input_prompt  = """
        You are an expert nutritionist. Identify each food item visible in the image
        and estimate its calorie content per serving. Use this format:

        1. **Food Item**: Number of calories per serving
        2. **Food Item**: Number of calories per serving

        Provide the total calories at the end.
        """
        response_text  = get_gemini_response(image_data, input_prompt)
        total_calories = calculate_total_calories(response_text)
        response_html  = markdown.markdown(response_text)
        formatted = (
            f"<h3>Calorie Breakdown:</h3>{response_html}"
            f"<p><strong>Total Calories: </strong>{total_calories}</p>"
        )
        return jsonify({"response": formatted})
    return jsonify({"error": "No image uploaded"}), 400


# ════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def agnos():   return render_template("index.html")

@app.route("/nutra")
def nutra():   return render_template("nutra.html")

@app.route("/chatbox")
def chatbox(): return render_template("chatbox.html")

@app.route("/login")
def login():   return render_template("Login.html")

@app.route("/signup")
def signup():  return render_template("signUp.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)