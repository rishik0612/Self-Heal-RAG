import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EvaluationResult:
    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_context: List[str]
    hallucinated: bool
    coverage: bool
    citations_present: bool
    retry_count: int
    grade: str

class HallucinationDetector:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY")
        )
    
    def detect(self, question: str, answer: str, context: List[str]) -> bool:
        """Returns True if answer contains hallucinations (claims not supported by context)."""
        context_str = "\n\n".join([f"Document {i+1}: {c}" for i, c in enumerate(context)]) if context else "No Documents"
        
        prompt = f"""You are a strict evaluator checking if an answer contains hallucinations (claims not supported by the provided context).

Context:
{context_str}

Question: {question}
Answer: {answer}

Does the answer make any factual claims that are NOT supported by the context?
Respond with ONLY: YES (contains unsupported claims) or NO (fully supported by context)."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            return "YES" in result
        except Exception as e:
            print(f"Hallucination detection error: {e}")
            return True  # Conservative: assume hallucination on error

class CoverageCalculator:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY")
        )
    
    def calculate(self, question: str, answer: str, ground_truth: str, context: List[str]) -> bool:
        """Returns True if answer adequately covers the ground truth given the context."""
        context_str = "\n\n".join([f"Document {i+1}: {c}" for i, c in enumerate(context)]) if context else "No Documents"
        
        prompt = f"""You are an evaluator checking if an answer adequately addresses the question given the available context.

Context:
{context_str}

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}

Does the answer adequately address the question using information available in the context?
Consider: If the context lacks information to answer the question, the answer should acknowledge this.
Respond with ONLY: YES (adequate coverage) or NO (inadequate coverage)."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            return "YES" in result
        except Exception as e:
            print(f"Coverage calculation error: {e}")
            return False

class CitationChecker:
    @staticmethod
    def check(answer: str) -> bool:
        """Check if answer contains source citations like [Source: filename.txt]."""
        import re
        pattern = r'\[Source:\s*[^\]]+\]'
        return bool(re.search(pattern, answer))

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data