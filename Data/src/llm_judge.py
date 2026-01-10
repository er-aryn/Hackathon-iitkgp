import ollama
from retrieval import NovelVectorStore
from pathlib import Path
import pandas as pd
import json

class LLMJudge:
    
    def __init__(self, vector_store, model_name="qwen2.5:3b"):
        self.vector_store = vector_store
        self.model_name = model_name
        print(f"Initialized LLM Judge with model: {model_name}")
    
    def extract_claims(self, backstory):
        
        sentences = [s.strip() + '.' for s in backstory.split('.') if len(s.strip()) > 30]
        return sentences[:5]
    
    def judge_consistency(self, backstory, book_name, caption=""):
        
        print(f"\n{'='*60}")
        print(f"Judging backstory for: {book_name}")
        print(f"{'='*60}")
        
        print("Extracting claims from backstory...")
        claims = self.extract_claims(backstory)
        print(f"Found {len(claims)} claims")
        
        print("Retrieving evidence from novel...")
        all_evidence = []
        for i, claim in enumerate(claims, 1):
            print(f"  Searching for claim {i}/{len(claims)}...")
            results = self.vector_store.search(
                query=claim,
                book_name=book_name,
                k=2  
            )
            all_evidence.extend(results)
        
        seen_chunks = set()
        unique_evidence = []
        for ev in sorted(all_evidence, key=lambda x: x['similarity'], reverse=True):
            chunk_key = (ev['book'], ev['chunk_id'])
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_evidence.append(ev)
                if len(unique_evidence) >= 5:
                    break
        
        print(f"Found {len(unique_evidence)} unique evidence passages")
        
        evidence_text = ""
        for i, ev in enumerate(unique_evidence, 1):
            evidence_text += f"\nPassage {i}:\n{ev['content'][:300]}...\n"
        
        print("Asking LLM to judge (this should take 10-15 seconds)...")
        judgment_prompt = f"""Backstory: {backstory[:400]}

        Novel passages:
        {evidence_text[:600]}

        Question: Is the backstory CONSISTENT and compatible with these novel passages?

        Answer in JSON:
        {{"prediction": 1, "reasoning": "brief reason"}}

        1 = consistent/compatible
        0 = contradicts"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": judgment_prompt}],
                options={
                    "temperature": 0.1,
                    "num_predict": 200
                }
            )
            
            print("LLM response received!")
            
            text = response['message']['content']
            print(f"Raw response: {text[:200]}...")
            
            text = text.replace('```json', '').replace('```', '').strip()
            result = json.loads(text)
            prediction = result['prediction']
            reasoning = result['reasoning']
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            prediction = 1
            reasoning = "Could not determine contradiction, defaulting to consistent"
        
        print(f"Prediction: {prediction} ({'CONSISTENT' if prediction == 1 else 'CONTRADICTORY'})")
        print(f"Reasoning: {reasoning}")
        
        return {
            'prediction': prediction,
            'reasoning': reasoning,
            'claims': claims,
            'evidence_count': len(unique_evidence)
        }


def test_llm_judge():
    print("="*60)
    print("TESTING LLM JUDGE")
    print("="*60)
    
    print("\n1. Loading vector store...")
    store = NovelVectorStore()
    
    if Path("Data/vector_store.pkl").exists():
        store.load()
    else:
        print("Error: Vector store not found! Run retrieval.py first.")
        return

    print("\n2. Creating LLM judge...")
    judge = LLMJudge(store)
    
    print("\n3. Loading test case...")
    train = pd.read_csv("Data/train-2.csv")
    sample = train.iloc[0]
    
    print(f"\nTest Case:")
    print(f"  Book: {sample['book_name']}")
    print(f"  True Label: {sample['label']}")
    print(f"  Backstory length: {len(sample['content'])} chars")
    
    book_map = {
        'In Search of the Castaways': 'castaways',
        'The Count of Monte Cristo': 'monte'
    }
    book_name = book_map.get(sample['book_name'], 'monte')
    
    print("\n4. Starting judgment process...")
    result = judge.judge_consistency(
        backstory=sample['content'],
        book_name=book_name,
        caption=sample['caption']
    )
    
    print(f"\n{'='*60}")
    print(f"RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"Predicted: {result['prediction']} ({'consistent' if result['prediction'] == 1 else 'contradict'})")
    print(f"Actual: {sample['label']}")
    correct = (result['prediction'] == 1 and sample['label'] == 'consistent') or \
              (result['prediction'] == 0 and sample['label'] == 'contradict')
    print(f"Match: {'CORRECT' if correct else 'WRONG'}")
    print(f"\nReasoning: {result['reasoning']}")


if __name__ == "__main__":
    test_llm_judge()