import sys 
# sys.path.append('/m:/All projects/RAG')
from model import Model
from evaluate import Evaluate
import os 
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
API_TOKEN = os.getenv("API_TOKEN")

def chat_mode():


    rag_model = Model(
        mongodb_url=MONGODB_URL, 
        mongodb_db=MONGODB_DB, 
        mongodb_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )
    query = input("Enter your question: ")
    answer = rag_model.get_answer(query)
    print("Answer:", answer)

def evaluate_mode():
    
    rag_model = Model(
        mongodb_url=MONGODB_URL, 
        mongodb_db=MONGODB_DB, 
        mongodb_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )
    evalaute = Evaluate()

    print("Evaluate True answer vs RAG answer")

    true_answer = input("Enter the true answer from textbook: ")

    rag_answer = rag_model.get_answer(true_answer)
    print("RAG Answer:", rag_answer)

    non_rag_answer = evalaute.get_non_rag_answer(true_answer)
    print("Non RAG Answer:", non_rag_answer)

    #Cosine Similarity Score to check RAG performance
    similarity_scores = eval.calculate_score(true_answer, rag_answer, non_rag_answer)
    print("RAG Answer: ",rag_answer,end="\n")
    print("Non Rag Answer: ",non_rag_answer,end="\n")
    print("Similarity Scores: ",end="\n")
    print("RAG Score: ",similarity_scores[0],end="\n")
    print("NON-RAG Score: ",similarity_scores[1],end="\n")

def main():
    print("Welcome to the RAG Chatbot! Type 1 for Chat Mode, Type 2 for Evaluate Mode: ")

    if input() == '1':
        chat_mode()
    else:
        evaluate_mode()
if __name__ == "__main__":
    main()