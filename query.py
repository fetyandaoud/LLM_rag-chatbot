from rag_core import ask_rag

chat_history = []


def print_help():
    print("\nKommandon:")
    print("- Skriv en fråga, t.ex. What is PRCO?")
    print("- Avsluta med: exit")
    print("- Hjälp: help\n")


def main():
    print_help()

    while True:
        question = input("Ask a question about your papers (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "help":
            print_help()
            continue

        chat_history.append({"role": "user", "content": question})
        result = ask_rag(question, chat_history[:-1])

        answer = result["answer"]
        sources = result["sources"]

        chat_history.append({"role": "assistant", "content": answer})

        print("\n" + "=" * 80)
        print("ANSWER:\n")
        print(answer)
        print("\nSOURCES:")
        for source in sources:
            print("-", source)
        print("=" * 80)


if __name__ == "__main__":
    main()