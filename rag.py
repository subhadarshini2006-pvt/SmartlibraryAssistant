from vector_store import search


def ask_question(question):
    """
    Retrieve the most relevant document and display the answer.
    """

    results = search(question)

    if len(results) == 0:
        print("\nNo relevant document found.")
        return

    print("\n======================================")
    print(" Most Relevant Document")
    print("======================================")

    print("File :", results[0]["file"])

    print("\nAnswer\n")

    print(results[0]["content"])


if __name__ == "__main__":

    print("========== Smart Library RAG ==========")

    while True:

        question = input("\nAsk a Question (type exit to quit): ")

        if question.lower() == "exit":
            break

        ask_question(question)