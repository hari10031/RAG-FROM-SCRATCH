import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class Evaluate:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def cosine_similarity(self,embedding1,embedding2):
        return 1 - cosine(embedding1,embedding2)
    
    def calcualte_score(self,true_answer,rag_answer,non_rag_answer):
        true_answer_embeddings = self.model.encode(true_answer,convert_to_tensor=False)
        rag_answer_embeddings = self.model.encode(rag_answer,convert_to_tensor=False)
        non_rag_answer_embeddings = self.model.encode(non_rag_answer,convert_to_tensor=False)
        rag_score = self.cosine_similarity(true_answer_embeddings,rag_answer_embeddings)
        non_rag_score = self.cosine_similarity(true_answer_embeddings,non_rag_answer_embeddings)
        return rag_score,non_rag_score