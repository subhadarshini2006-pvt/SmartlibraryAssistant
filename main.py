from cover_reader import read_cover
from rag import ask_question
from demand_prediction import predict_book
from vector_store import build_vector_database
import os


def menu():

    while True:

        print("\n" + "="*50)
        print("      SMART LIBRARY ASSISTANT")
        print("="*50)

        print("1. Read Book Cover (Computer Vision)")
        print("2. Ask Question (RAG)")
        print("3. Predict High Demand Book (ML)")
        print("4. Build Vector Database")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        # -------------------------------
        # OCR
        # -------------------------------

        if choice == "1":

            image_path = input("\nEnter image path (default:data/cover.jpg): ")

            if image_path == "":
                image_path = "data/cover.jpg"

            if not os.path.exists(image_path):

                print("\nImage not found!")

                continue

            text = read_cover(image_path)

            print("\nDetected Text")
            print("-"*40)
            print(text)

        # -------------------------------
        # RAG
        # -------------------------------

        elif choice == "2":

            question = input("\nAsk your question: ")

            ask_question(question)

        # -------------------------------
        # ML Prediction
        # -------------------------------

        elif choice == "3":

            month = input("\nEnter Month: ")

            result = predict_book(month)

            print("\nPredicted High Demand Book")

            print("----------------------------")

            print(result)

        # -------------------------------
        # Build Vector DB
        # -------------------------------

        elif choice == "4":

            build_vector_database()

        # -------------------------------
        # Exit
        # -------------------------------

        elif choice == "5":

            print("\nThank you for using Smart Library Assistant.")

            break

        else:

            print("\nInvalid Choice.")


if __name__ == "__main__":

    menu()