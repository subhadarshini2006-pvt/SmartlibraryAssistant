import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENT_FOLDER = "data/documents"

VECTOR_FOLDER = "vector_db"

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
DOC_FILE = os.path.join(VECTOR_FOLDER, "documents.pkl")


def load_documents():

    documents = []
    file_names = []

    if not os.path.exists(DOCUMENT_FOLDER):
        print("Document folder not found.")
        return documents, file_names

    for file in os.listdir(DOCUMENT_FOLDER):

        if file.endswith(".txt"):

            path = os.path.join(DOCUMENT_FOLDER, file)

            with open(path, "r", encoding="utf-8") as f:

                text = f.read()

                documents.append(text)

                file_names.append(file)

    return documents, file_names


def build_vector_database():

    documents, file_names = load_documents()

    if len(documents) == 0:
        print("No documents found.")
        return

    print("Creating embeddings...")

    embeddings = model.encode(documents)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(DOC_FILE, "wb") as f:

        pickle.dump(
            {
                "documents": documents,
                "files": file_names
            },
            f
        )

    print("\nVector Database Created Successfully!")

    print("Total Documents :", len(documents))


def search(query, k=1):

    if not os.path.exists(INDEX_FILE):

        print("Vector database not found.")

        print("Run build_vector_database() first.")

        return []

    index = faiss.read_index(INDEX_FILE)

    with open(DOC_FILE, "rb") as f:

        data = pickle.load(f)

    documents = data["documents"]

    file_names = data["files"]

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:

        results.append(
            {
                "file": file_names[i],
                "content": documents[i]
            }
        )

    return results


if __name__ == "__main__":

    build_vector_database()

    print("\nTesting Search\n")

    query = input("Enter your question: ")

    results = search(query)

    print("\nMost Relevant Document\n")

    print("----------------------------")

    print(results[0]["file"])

    print()

    print(results[0]["content"])