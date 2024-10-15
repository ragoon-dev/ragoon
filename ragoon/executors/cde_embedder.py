import abc
import random
import string

from datasets import Dataset

from ragoon.executors.embedder import BaseEmbedder
from ragoon.models.base import Config


class CDEEmbedder(BaseEmbedder):
    def __init__(self, config: Config):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
        self.embedding_limit = config.embed.training_size_limit
        self.input_feature = config.training_data.input_feature
        self.docs_embedding_count = config.embed.docs_embedding_count
        self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.create_collection(
            name=self.normalize_name(config.name),
        )

    def normalize_name(self, collection_name: str):
        return collection_name.lower().replace(" ", "-").replace("/", "-").replace(
            ":", "-"
        )[:32] + "".join(random.choices(string.ascii_letters, k=5))

    def set_training_dataset(self, training_dataset: Dataset):
        self.train_dataset = training_dataset
        self.max_range = (
            self.embedding_limit
            if self.embedding_limit is not None and int(self.embedding_limit) > 0
            else len(self.train_dataset[self.input_feature])
        )

    def embedd(self):
        self.training_corpus = self.train_dataset[self.input_feature][: self.max_range]
        self.docs_embedding_count = (
            self.docs_embedding_count
            if self.docs_embedding_count < self.max_range
            else self.max_range
        )
        self.dataset_embeddings = self.model.encode(
            random.sample(self.training_corpus, k=self.docs_embedding_count),
            prompt_name="document",
            convert_to_tensor=True,
        )

        doc_embeddings = self.model.encode(
            self.training_corpus,
            prompt_name="document",
            dataset_embeddings=self.dataset_embeddings,
            convert_to_tensor=True,
        ).cpu()
        self.collection.add(
            embeddings=doc_embeddings,
            ids=list(map(lambda x: str(x), range(0, self.max_range))),
        )

    def get_similar(self, query: str, k: int):
        query_embedding = self.model.encode(
            query,
            prompt_name="query",
            dataset_embeddings=self.dataset_embeddings,
            convert_to_tensor=True,
        ).cpu()
        ids = self.collection.query(query_embeddings=[query_embedding], n_results=k)[
            "ids"
        ][0]
        documents = [self.training_corpus[int(i)] for i in ids]
        return {"ids": [ids], "documents": [documents]}
