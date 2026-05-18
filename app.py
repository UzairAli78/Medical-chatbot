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


# embeddings = download_hugging_face_embeddings() 
# OR load the embeddings 
embeddings = joblib.load("embeddings.joblib")        #---- Look Here before execution----

#Initializing the Pinecone
index_name="medical-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

#Loading the BM25encodings
bm25_encoder = BM25Encoder().load("bm25_values.joblib")
# bm25_encoder = joblib.load("bm25_values.joblib")

#Loading the index
retriever = PineconeHybridSearchRetriever(embeddings=embeddings,
                                          sparse_encoder=bm25_encoder,
                                           index=index)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,
                            model="gemini-2.5-flash",api_version="v1", name="gemini",
                            temperature=0.8
                            )

# Assuming 'retriever' is an instance of a concrete retriever class
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,  # Pass the retriever instance directly
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    
    # Invoke the model or QA system
    result = qa.invoke({"query": input})
    
    # Extract the result from the response
    response_text = result.get("result", "")
    print("Response : ", response_text)
    
    # Convert markdown-like syntax to proper HTML
    response_html = markdown.markdown(response_text)

    # Return the formatted HTML response
    return response_html


# -------------------------------------------------------------------------------------------------------

def get_gemini_response(image_blob, prompt):
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    # Call the API with the prompt and image data
    response = model.generate_content([prompt, image_blob])
    return response.text

# Function to handle image setup
def input_image_setup(uploaded_file):
    if uploaded_file:
        # Read the file into bytes
        bytes_data = uploaded_file.read()

        # Convert the bytes data to a PIL Image
        image = Image.open(io.BytesIO(bytes_data))

        # Prepare the image as a dict (Blob)
        image_blob = {
            "mime_type": uploaded_file.content_type,
            "data": bytes_data
        }
        return image_blob
    else:
        raise FileNotFoundError("No file uploaded")
    
def calculate_total_calories(response_text):
    """
    Function to calculate the total calories from the response text.
    The response text is expected to contain food items with calorie counts in the format:
    1. Food Item: XX calories
    2. Food Item: YY calories
    """
    # Use regex to find all occurrences of calorie values (assumed to be numbers)
    calorie_values = re.findall(r'(\d+)\s*calories', response_text)

    # Convert the extracted calorie values to integers and sum them
    total_calories = sum(map(int, calorie_values))

    return total_calories    


import markdown

@app.route('/calculate_calories', methods=['POST'])
def calculate_calories():
    input_text = request.form.get('input')
    uploaded_file = request.files.get('image')

    if uploaded_file:
        # Prepare image data for Google Gemini API
        image_data = input_image_setup(uploaded_file)

        # Input prompt for the nutritionist task
        input_prompt = """
        You are an expert nutritionist where you need to see the food items from the image
        and calculate the total calories. Provide the details of each food item with its calorie intake 
        in the following format:

        1. **Food Item**: Number of calories per serving
        2. **Food Item**: Number of calories per serving

        Provide the total calories at the end of the list.
        """

        # Call Gemini API and get response
        response_text = get_gemini_response(image_data, input_prompt)

        # Calculate total calories from the response text
        total_calories = calculate_total_calories(response_text)

        # Use the markdown package to convert any markdown-like syntax to HTML
        response_text_html = markdown.markdown(response_text)

        # Format response into HTML for better readability
        formatted_response = f"""
        <h3>Calorie Breakdown:</h3>
        {response_text_html}
        <p><strong>Total Calories: </strong> {total_calories}</p>
        """

        # Send the formatted response back to the client
        return jsonify({"response": formatted_response})
    
    return jsonify({"error": "No image uploaded"}), 400





# Route for the homepage
@app.route('/')
def agnos():
    return render_template('index.html')

# Route for the Nutra page
@app.route('/nutra')
def nutra():
    return render_template('nutra.html')

# Route for the Chatbot page
@app.route("/chatbox")
def chatbox():
    return render_template('chatbox.html')

# Route for Login page
@app.route("/login")
def login():
    return render_template('Login.html')

# Route for Sign Up page
@app.route("/signup")
def signup():
    return render_template('signUp.html')





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


