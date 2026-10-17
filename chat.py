from services.ragservice import RAGService


def main() -> None:
    """Run a one-off question from the terminal."""
    service = RAGService()
    question = input("Question: ").strip()
    if question:
        print(service.query(question))


if __name__ == "__main__":
    main()