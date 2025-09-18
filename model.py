from sentence_transformers import SentenceTransformer
import pymongo
import numpy as np 
from scipy.spatial.distance import cosine
import os
import google.generativeai as genai
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class Model:
    def __init__(self,mongodb_url,mongodb_db,mongodb_collection,api_token):
        self.mongodb_url = mongodb_url
        self.mongodb_db = mongodb_db
        self.mongodb_collection = mongodb_collection
        self.api_token = api_token
        self.client = pymongo.MongoClient(self.mongodb_url)
        self.db= self.client[self.mongodb_db]
        self.collection = self.db[self.mongodb_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        def semantic_search(self,query,top_k=5):
            query_emebedding = self.model.encode(query,convert_to_tensor=False)
            similarity = []
            for doc in self.collection.find():
                doc_embedding = np.array(doc['embedding'])
                similar = 1 - cosine(query_emebedding,doc_embedding)
                similarity.append((doc['_id'],similar,doc['text']))
            similarity.sort(key=lambda x: x[1],reverse=True)
            return similarity[:top_k]
        
        def get_answer(self,question,max_content_length=500):

            doc_search = self.semantic_search(question,top_k=1)
            if doc_search:
                context = doc_search[0][2]
                if len(context) > max_content_length:
                    context = context[:max_content_length]
                prompt = (
                    "You are a helpful assistant. Use the provided context to answer the question.\n\n"
                    f"Question: {question}\n"
                    f"Context: {context}\n\n"
                    "Answer:"
                )
            else:
                prompt = (
                    "You are a helpful assistant. Answer the question to the best of your ability.\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )
            genai.configure(api_key=self.api_token)
            response = genai.generate_content(
                model="gemini-2.5-pro",
                contents=[{"role": "user", "parts": [prompt]}],
                generation_config={
                    "temperature": 0.5,
                    "topK": 20,
                    "topP": 1.0,
                    "maxOutputTokens": 1024
                }
            )
            answer = ""
            print(response.text)
            for item in response.text:
                answer += item
            
            if not answer:
                answer = "Sorry, I dont have an answer for that!!!"
            return answer
        
        def get_rag_answer(self,question,max_context_length=500):

            doc_search = self.semantic_search(question,top_k=1)
            context = doc_search[0][2]
            if len(context) > max_context_length:
                context = context[:max_context_length]
            prompt = (
                "You are a helpful assistant. Use the provided context to answer the question.\n\n"
                f"Question: {question}\n"
                f"Context: {context}\n\n"
                "Answer:"
            )
            response = genai.generate_content(
                model="gemini-2.5-pro",
                contents=[{"role": "user", "parts": [prompt]}],
                generation_config={
                    "temperature": 0.5,
                    "topK": 20,
                    "topP": 1.0,
                    "maxOutputTokens": 1024
                }
            )
            answer = ""
            for item in response.text:
                answer += item
            if not answer:
                answer = "Sorry, I dont have an answer for that!!!"
            return answer
        
        def get_non_rag_answer(self,question):
            prompt = (
                "You are a helpful assistant. Use the provided context to answer the question.\n\n"
                f"Question: {question}\n"
            )
            response = genai.generate_content(
                model="gemini-2.5-pro",
                contents=[{"role": "user", "parts": [prompt]}],
                generation_config={
                    "temperature": 0.5,
                    "topK": 20,
                    "topP": 1.0,
                    "maxOutputTokens": 1024
                }
            )
            ans = ""
            for item in response.text:
                ans += item
            if not ans:
                ans = "Sorry, I dont have an answer for that!!!"
            return ans
                    
                



