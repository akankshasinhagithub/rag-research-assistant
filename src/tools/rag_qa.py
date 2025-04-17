import os
from openai import OpenAI
from search_faiss import search_index, get_results


def generate_answer(context, question):
    prompt = f"""You are a helpful research assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {question}
    """
    response = OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    query = "What is RAG? Why it is important to the AI community?"

    # Dynamically resolve path to FAISS index file (from current file location)
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../my_index.faiss"))

    # Search index and retrieve relevant chunks
    distances, indices = search_index(query, index_path=index_path)
    chunks = get_results(indices)
    context = "\n".join(chunks)

    # Generate and print the answer
    answer = generate_answer(context, query)
    print("\n Final Answer:\n", answer)
