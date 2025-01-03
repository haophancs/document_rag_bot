import cloudpickle
import joblib
import pandas as pd
import stopwordsiso
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    """TF-IDF Embeddings."""

    def __init__(self, vectorizer_path: str) -> None:
        super().__init__()
        self.vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed search docs.

        :param texts: List of text to embed.

        :returns: List of embeddings.
        """
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text.

        :param text: Text to embed.

        :returns: Embedding.
        """
        return self.vectorizer.transform([text]).toarray()[0].tolist()


def custom_tokenizer(text: str) -> list[str]:
    """Custom tokenizer for tfidf vectorizer."""
    import re  # fmt: off

    pattern = r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+"
    return re.findall(pattern, re.sub(r"[_\-\s]", " ", text))


if __name__ == "__main__":
    corpus = pd.read_csv("./assets/documents.csv")["text"].tolist()

    english_stop_words = set(stopwordsiso.stopwords("en"))
    vietnamese_stop_words = set(stopwordsiso.stopwords("vi"))
    all_stop_words = list(english_stop_words.union(vietnamese_stop_words))

    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        stop_words=all_stop_words,
        token_pattern=None,
    )
    vectorizer.fit(corpus)
    with open("./assets/tfidf_vectorizer.joblib", mode="wb") as file:
        cloudpickle.dump(vectorizer, file)
