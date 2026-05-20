from flask import Flask, render_template, jsonify, request
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
from dotenv import load_dotenv
import pinecone
import joblib
import markdown
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os
from PIL import Image
import io
import re


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Load embeddings
embeddings = joblib.load("embeddings.joblib")

# Initializing the Pinecone
index_name = "medical-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Loading the BM25 encodings
bm25_encoder = BM25Encoder().load("bm25_values.joblib")

# Loading the retriever
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
    temperature=0.8
)

# RAG pipeline for medical questions
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Direct Gemini model for casual conversation
casual_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=(
        "You are MediChat AI, a friendly and warm medical assistant chatbot. "
        "You have a natural, conversational personality. When users make small talk or greet you, "
        "respond naturally and warmly, then gently invite them to ask a medical question if they have one. "
        "You can chat casually but always keep in mind your main purpose is to help with medical questions. "
        "Keep responses concise and friendly."
    )
)


# ── Smart router ──────────────────────────────────────────────────────────────

# Keywords that suggest a medical question — route to RAG
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
    "patient", "physician", "specialist", "what is", "how to treat",
    "how do i", "should i take", "can i take", "is it safe", "side effect",
    "can you prescribe", "what causes", "why do i", "i have", "i feel",
    "i am experiencing", "my", "ache", "sore", "inflammation"
]

# Phrases that are clearly casual / greetings — route to direct Gemini
CASUAL_PHRASES = [
    "hello", "hi", "hey", "hii", "helo", "good morning", "good afternoon",
    "good evening", "how are you", "what's up", "whats up", "sup",
    "who are you", "what can you do", "tell me about yourself",
    "what are you", "nice to meet", "thanks", "thank you", "ok", "okay",
    "cool", "great", "awesome", "bye", "goodbye", "see you", "later"
]


def is_casual(message: str) -> bool:
    """Return True if the message looks like casual chat rather than a medical question."""
    msg = message.lower().strip()

    # If it matches a casual phrase and has no medical keywords, it's casual
    for phrase in CASUAL_PHRASES:
        if msg == phrase or msg.startswith(phrase + " ") or msg.startswith(phrase + "!") or msg.startswith(phrase + ","):
            if not any(kw in msg for kw in MEDICAL_KEYWORDS):
                return True

    # If any medical keyword is present, route to RAG
    if any(kw in msg for kw in MEDICAL_KEYWORDS):
        return False

    # Short messages with no medical keywords → casual
    if len(msg.split()) <= 5:
        return True

    return False


# ── Chat route ────────────────────────────────────────────────────────────────

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")

    if is_casual(msg):
        # Handle casual conversation directly with Gemini (no RAG)
        try:
            response = casual_model.generate_content(msg)
            response_text = response.text
        except Exception as e:
            print(f"Casual model error: {e}")
            response_text = "Hey there! I'm MediChat AI, your medical assistant. Feel free to ask me any medical question and I'll do my best to help!"
    else:
        # Handle medical questions through the RAG pipeline
        try:
            result = qa.invoke({"query": msg})
            response_text = result.get("result", "")
        except Exception as e:
            print(f"RAG error: {e}")
            response_text = "I'm sorry, I had trouble processing that. Could you rephrase your question?"

    print(f"Response: {response_text}")
    response_html = markdown.markdown(response_text)
    return response_html


# ── Calorie estimator ─────────────────────────────────────────────────────────

def get_gemini_response(image_blob, prompt):
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content([prompt, image_blob])
    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        image_blob = {
            "mime_type": uploaded_file.content_type,
            "data": bytes_data
        }
        return image_blob
    else:
        raise FileNotFoundError("No file uploaded")


def calculate_total_calories(response_text):
    calorie_values = re.findall(r'(\d+)\s*calories', response_text)
    total_calories = sum(map(int, calorie_values))
    return total_calories


@app.route('/calculate_calories', methods=['POST'])
def calculate_calories():
    input_text = request.form.get('input')
    uploaded_file = request.files.get('image')

    if uploaded_file:
        image_data = input_image_setup(uploaded_file)

        input_prompt = """
        You are an expert nutritionist where you need to see the food items from the image
        and calculate the total calories. Provide the details of each food item with its calorie intake 
        in the following format:

        1. **Food Item**: Number of calories per serving
        2. **Food Item**: Number of calories per serving

        Provide the total calories at the end of the list.
        """

        response_text = get_gemini_response(image_data, input_prompt)
        total_calories = calculate_total_calories(response_text)
        response_text_html = markdown.markdown(response_text)

        formatted_response = f"""
        <h3>Calorie Breakdown:</h3>
        {response_text_html}
        <p><strong>Total Calories: </strong> {total_calories}</p>
        """

        return jsonify({"response": formatted_response})

    return jsonify({"error": "No image uploaded"}), 400


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route('/')
def agnos():
    return render_template('index.html')

@app.route('/nutra')
def nutra():
    return render_template('nutra.html')

@app.route("/chatbox")
def chatbox():
    return render_template('chatbox.html')

@app.route("/login")
def login():
    return render_template('Login.html')

@app.route("/signup")
def signup():
    return render_template('signUp.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)