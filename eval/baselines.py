from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class BaselineResult:
    variant_name: str
    question: str
    answer: str
    retrieved_context: List[str]
    retry_count: int
    grade: str
    hallucinated: bool
    coverage: bool
    citations_present: bool

def create_single_pass_graph(max_retries: int = 0):
    """Create a graph with no self-healing (single pass only)."""
    from langgraph.graph import StateGraph, END
    from ragagent import retrieve, generate, grade_answer, give_up, AgentState
    
    def no_retry_route(state: AgentState) -> str:
        grade = state.get("grade", "fail")
        if grade == "pass":
            return "end"
        else:
            return "give_up"
    
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("give_up", give_up)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_edge("give_up", END)
    graph.add_conditional_edges(
        "grade_answer",
        no_retry_route,
        {"end": END, "give_up": "give_up"}
    )
    return graph.compile()

def create_fixed_retry_graph(max_retries: int):
    """Create a graph with fixed number of retries."""
    from langgraph.graph import StateGraph, END
    from ragagent import retrieve, generate, grade_answer, rewrite_question, give_up, AgentState, route_after_grading
    
    def fixed_retry_route(state: AgentState) -> str:
        grade = state.get("grade", "fail")
        retry_count = state.get("retry_count", 0)
        if grade == "pass":
            return "end"
        elif retry_count >= max_retries:
            return "give_up"
        else:
            return "rewrite"
    
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
    graph.add_conditional_edges(
        "grade_answer",
        fixed_retry_route,
        {"end": END, "give_up": "give_up", "rewrite": "rewrite"}
    )
    return graph.compile()

def run_baseline(graph, question: str, variant_name: str, hallucination_detector, coverage_calculator, citation_checker) -> BaselineResult:
    """Run a single question through a baseline graph."""
    from ragagent import AgentState
    
    initial_state: AgentState = {
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "answer": "",
        "grade": "",
        "retry_count": 0
    }
    
    final_state = graph.invoke(initial_state)
    
    # Extract context
    context = []
    for doc in final_state.get("documents", []):
        if isinstance(doc, dict):
            context.append(doc.get("content", ""))
        else:
            context.append(str(doc))
    
    answer = final_state.get("answer", "")
    retry_count = final_state.get("retry_count", 0)
    grade = final_state.get("grade", "fail")
    
    # Evaluate
    hallucinated = hallucination_detector.detect(question, answer, context)
    coverage = coverage_calculator.calculate(question, answer, "", context)  # Ground truth not needed for coverage
    citations = citation_checker.check(answer)
    
    return BaselineResult(
        variant_name=variant_name,
        question=question,
        answer=answer,
        retrieved_context=context,
        retry_count=retry_count,
        grade=grade,
        hallucinated=hallucinated,
        coverage=coverage,
        citations_present=citations
    )

def get_all_baselines():
    """Return all baseline configurations to test."""
    return [
        ("single_pass", create_single_pass_graph()),
        ("one_retry", create_fixed_retry_graph(1)),
        ("two_retry", create_fixed_retry_graph(2)),
    ]