import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import io
import requests
from tqdm import tqdm

from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.databases import FaissIndex
from src.models import VllmEmbedding


def get_text_from_url(url: str):
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        if url.endswith(".pdf"):
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = "".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        else:
            soup = BeautifulSoup(
                response.content,
                "html.parser",
                from_encoding="utf-8",
            )
            text = " ".join(p.get_text() for p in soup.find_all("p"))

        return text.strip() if text else None
    except Exception:
        return None


@hydra.main(
    config_path="configs/",
    config_name="main.yaml",
)
def set_vector_store(
    config: DictConfig,
) -> None:
    index: FaissIndex = instantiate(
        config.database,
    )
    embedding: VllmEmbedding = instantiate(
        config.model.embedding,
    )

    documents_path = os.path.join(
        config.data_path,
        config.documents_name,
    )
    docs_df = pd.read_csv(documents_path)

    tqdm.pandas(desc="Fetching text from URLs")
    docs_df["text"] = docs_df["url"].progress_apply(get_text_from_url)
    docs_df.dropna(subset=["text"], inplace=True)
    docs_df = docs_df[docs_df["text"] != ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = []
    for _, row in tqdm(
        docs_df.iterrows(), total=docs_df.shape[0], desc="Splitting text"
    ):
        for chunk_text in text_splitter.split_text(row["text"]):
            chunks.append(
                {
                    "domain": row["domain"],
                    "file_name": row["file_name"],
                    "url": row["url"],
                    "chunk": chunk_text,
                }
            )

    chunks_df = pd.DataFrame(chunks)

    queries = chunks_df["chunk"].tolist()
    embedded = [
        embedding(query=query) for query in tqdm(queries, desc="Generating embeddings")
    ]
    embedded = np.array(
        embedded,
        dtype=np.float32,
    )

    index.add(embedded=embedded)
    index.df = chunks_df
    index.save()


if __name__ == "__main__":
    set_vector_store()
