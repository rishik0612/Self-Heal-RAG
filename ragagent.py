import os 
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
load_dotenv()
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[dict]
    answer: str
    grade: str
    retry_count: int

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base", 
    metadata={"hnsw:space": "cosine"}
)
def retrieve(state: AgentState) -> AgentState:
    """Retrieves relevant documents and their source metadata from ChromaDB based on the current question."""
    question = state.get("rewritten_question") or state["question"]
    print(f"Retrieving documents for question: {question}")
    try:
        results = collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "metadatas"]
        )
        doc_texts = results.get("documents", [[]])[0]
        doc_metas = results.get("metadatas", [[]])[0]
        documents = []
        for text, meta in zip(doc_texts, doc_metas):
            source = meta.get("source", "Unknown Source") if meta else "Unknown Source"
            documents.append({"content": text, "source": source})
        print(f"Retrieved {len(documents)} documents from ChromaDB.")
    except Exception as error:
        print(f"Error during retrieval: {error}")
        documents = []
    return {**state, "documents": documents}

def generate(state: AgentState) -> AgentState:
    """Generates an answer with citations based on the retrieved documents and the question."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    print(f"Generating answer for question: {question}")
    if not documents:
        return {**state, "answer": "No relevant documents found to answer the question."}
    
    context_blocks = []
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            context_blocks.append(f"Document {i+1} [Source: {doc.get('source', 'Unknown')}]:\n{doc.get('content', '')}")
        else:
            context_blocks.append(f"Document {i+1}:\n{doc}")
            
    context = "\n\n".join(context_blocks)
    system_prompt = f"""You are a helpful assistant that answers questions based on retrieved documents.
ALWAYS cite the source file name (e.g., [Source: filename.txt]) for every factual claim or piece of information used in your answer.

Retrieved Documents:
{context}"""

    user_prompt = f"""Question: {question}

Answer the question based on the retrieved documents. Ensure you include inline citations citing the source document(s) (e.g., [Source: filename.txt])."""

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        answer = response.content.strip()
        print(f"[GENERATE] Generated answer: len({len(answer)}) characters")
    except Exception as error:
        print(f"[GENERATE] LLM call failed: {error}")
        answer = "An error occurred while generating answer."
    return {**state, "answer": answer}

def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents and includes source citations."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")
    print(f"\n[GRADE] evaluating answer quality...")
    
    context_blocks = []
    for d in documents:
        if isinstance(d, dict):
            context_blocks.append(f"Source: {d.get('source', 'Unknown')}\nContent: {d.get('content', '')}")
        else:
            context_blocks.append(str(d))
            
    context = "\n\n".join(context_blocks) if context_blocks else "No Documents"
    grading_prompt = f"""You are a strict grader evaluating whether an answer is grounded in the provided documents and includes source citations.
    Documents: 
    {context}
    
    Question: {question}
    Answer: {answer}    
    
    Evaluate this answer on three criteria:
    1. Is the answer directly supported by information in the documents?
    2. Does the answer avoid making claims not found in the documents?
    3. Does the answer cite the source document(s) (e.g., [Source: filename.txt])?

    Respond with ONLY one word: PASS or FAIL.
    PASS means the answer is well-supported and includes source citations.
    FAIL means the answer contains unsupported claims, lacks citations, or is not relevant."""
    try:
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        grade_raw = response.content.strip().upper()
        grade = "pass" if "PASS" in grade_raw else "fail"
        print(f"[GRADE] Result: {grade.upper()}")
    except Exception as error:
        print(f"[GRADE] Grading call failed: {error}")
        grade = "fail"
    return {**state, "grade": grade}

def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval on the nextattempt."""
    original_question = state["question"]
    retry_count = state.get("retry_count", 0)
    print(f"\n[REWRITE] Rewriting question (attempt {retry_count + 1})...")
    rewrite_prompt = f"""The following question did not retrieve useful documents 
    from a knowledge base. Rewrite the question to be more specific and use
    different vocabulary that might match relevant documents better.
    Original question: {original_question}
    Write only the rewritten question, nothing else:"""
    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()
        print(f"[REWRITE] New question: {rewritten}")
    except Exception as error:
        print(f"[REWRITE] Rewrite failed: {error}")
        rewritten = original_question
    return {**state, "rewritten_question": rewritten, "retry_count": retry_count + 1}

def give_up(state: AgentState) -> AgentState:
    """Return an honest 'I don't know' after all retries are exhausted."""
    print(f"\n[GIVE UP] Could not find a reliable answer after retries.")
    honest_answer = ("I was unable to find a reliable answer to your question in the available documents.\n The knowledge base may not contain information about this topic.\n Please try rephrasing your question \n or consult additional sources.")
    return {**state, "answer": honest_answer, "grade": "fail"}

def route_after_grading(state: AgentState) -> str:
    """Decide where to go after grading the answer."""
    grade = state.get("grade", "fail")
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    if grade == "pass":
        return "end"
    elif retry_count >= max_retries:
        return "give_up"
    else:
        return "rewrite"
    
def build_graph() -> StateGraph:
    """Assemble all nodes and edges into the LangGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite", rewrite_question)
    graph.add_node("give_up", give_up)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("give_up", END)
    # Add the conditional edge after grading
    graph.add_conditional_edges(
        "grade_answer",
        route_after_grading,
        {
            "end": END,
            "give_up": "give_up",
            "rewrite": "rewrite"
        }
    )
    return graph.compile()

if __name__ == "__main__":
    print("Self-Healing RAG System")
    print("=" * 40)
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.\n")
    rag_graph = build_graph()
    while True: 
        user_question = input("Your question: ").strip()
        if user_question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_question:
            continue
        initial_state: AgentState = {
            "question": user_question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0
            }
        try:
            final_state = rag_graph.invoke(initial_state)
            print(f"\n[FINAL ANSWER] {final_state['answer']}")
        except Exception as error:
            print(f"Error executing RAG graph: {error}")