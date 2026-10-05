import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
class SenticseEmbeddings:
    def __init__(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

        self.similarity_fn_name = "cosine"

        self.model_card_data = SentenceTransformerModelCardData()

    def encode(self, texts, **kwargs):
        features = []
        self.model.eval()

        with torch.no_grad():
            for text in tqdm(texts):
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                output = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                ).pooler_output
                features.append(output.cpu().numpy())

        embeddings = np.vstack(features)
        
        # Normalize only if flag is True
        if kwargs.get('normalize_embeddings', False):
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings

    def encode_query(self, texts, **kwargs):
            return self.encode(texts, **kwargs)

    def encode_document(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def similarity(self, embeddings1, embeddings2):
        embeddings1 = torch.as_tensor(embeddings1)
        embeddings2 = torch.as_tensor(embeddings2)

        embeddings1 = torch.nn.functional.normalize(
            embeddings1, p=2, dim=-1
        )
        embeddings2 = torch.nn.functional.normalize(
            embeddings2, p=2, dim=-1
        )

        return embeddings1 @ embeddings2.T