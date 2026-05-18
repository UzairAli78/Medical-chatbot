# 🏥 Medical Chatbot

An AI-powered medical chatbot built with Flask, Google Gemini, Pinecone, and LangChain. It answers medical questions using a hybrid search approach over a medical knowledge base, and also includes a calorie estimation feature from food images.

---

## ✨ Features

- **Medical Q&A** — Ask any medical question and get accurate, context-aware answers sourced from a medical knowledge base
- **Hybrid Search** — Combines dense (HuggingFace) and sparse (BM25) vector search via Pinecone for highly relevant retrieval
- **Diagnostic Conversation** — The chatbot engages with follow-up questions to better understand your concern
- **Calorie Estimator** — Upload a food image and get a detailed calorie breakdown powered by Google Gemini Vision
- **Responsive UI** — Clean, modern web interface built with Flask templates and Bootstrap

---

## 🛠️ Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| Backend         | Python, Flask                           |
| LLM             | Google Gemini 2.5 Flash                 |
| Embeddings      | HuggingFace `all-MiniLM-L6-v2`          |
| Sparse Encoding | BM25Encoder (pinecone-text)             |
| Vector Store    | Pinecone (Serverless, AWS us-east-1)    |
| Retrieval       | LangChain PineconeHybridSearchRetriever |
| Frontend        | HTML, CSS, Bootstrap, jQuery            |

---

## 📁 Project Structure

```
medical-chatbot/
├── app.py                  # Main Flask application
├── store_index.py          # One-time script to embed PDF and upload to Pinecone
├── src/
│   ├── helper.py           # PDF loading, text splitting, embedding utilities
│   └── prompt.py           # LLM prompt template
├── templates/              # HTML templates
├── static/                 # CSS, JS, images, fonts
├── data/
│   └── Medical_book.pdf    # Medical knowledge base (not included in repo)
├── environment.yml         # Conda environment definition
├── requirements.txt        # Full pip dependencies
└── .env                    # API keys (not committed)
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10
- A [Pinecone](https://app.pinecone.io) account and API key
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key

---

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-chatbot.git
cd medical-chatbot
```

### 2. Create a virtual environment

**Option A — Using venv (Python 3.10 required):**

```bash
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Option B — Using Conda:**

```bash
conda env create -f environment.yml
conda activate med_chatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_API_KEY_ENV=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_API_KEY_ENV=your_google_api_key_here
```

### 5. Create the Pinecone index

Log in to [Pinecone](https://app.pinecone.io) and create an index with these settings:

| Setting       | Value             |
| ------------- | ----------------- |
| Index name    | `medical-chatbot` |
| Vector type   | Dense             |
| Dimensions    | `384`             |
| Metric        | `dotproduct`      |
| Capacity mode | Serverless        |
| Cloud         | AWS               |
| Region        | us-east-1         |

### 6. Embed and upload the knowledge base

> Run this only once. It reads the medical PDF, generates embeddings, and uploads vectors to Pinecone.

```bash
python store_index.py
```

This will take 10–20 minutes depending on your machine. You'll see this message when it's done:

```
Hybrid vectors are available in the vectorstore Pinecone
```

### 7. Run the application

```bash
python app.py
```

Open your browser at **http://localhost:8080**

---

## 🔑 Environment Variables

| Variable               | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `PINECONE_API_KEY`     | Your Pinecone API key (used by `app.py`)              |
| `PINECONE_API_KEY_ENV` | Your Pinecone API key (used by `store_index.py`)      |
| `GOOGLE_API_KEY`       | Your Google Gemini API key (used by `app.py`)         |
| `GOOGLE_API_KEY_ENV`   | Your Google Gemini API key (used by `store_index.py`) |

---

## 🚀 Usage

- Navigate to `http://localhost:8080`
- Use the **Chatbot** page to ask medical questions
- Use the **Calorie Estimator** to upload a food image and get a nutritional breakdown

---

## ⚠️ Disclaimer

This chatbot is for **informational and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

This project is licensed under the terms included in the `LICENSE` file.
