import streamlit as st
import os 
from dotenv import load_dotenv

from model import Model
from evaluate import Evaluate

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
API_TOKEN = os.getenv("API_TOKEN")

@st.cache(allow_output_mutation=True)
def load_model():
    return  Model(
        mongodb_url=MONGODB_URL, 
        mongodb_db=MONGODB_DB, 
        mongodb_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

st.set_page_config(page_title="RAG ChatBot")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       .reportview-container {
            background: #091c23
       }
       
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #00b8a9;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    </style>


    """,unsafe_allow_html=True
)

footer = """
    <style>
    .reportview-container .main footer {
        visibility: hidden;
    }    
    </style>
    """

st.markdown(footer, unsafe_allow_html=True)

st.title("RAG ChatBot")
st.write("Welcome to the RAG ChatBot! Type your question below and get answers based on the context from the database.")


query = st.text_input("Enter your question here:")

model = load_model()

if query:
    st.write("Please Hold a moment, fetching the answer...")
answer = model.get_answer(query)
st.write("Answer:", answer)
