# src/pipeline.py

import pandas as pd
from pathlib import Path
from retrieval import NovelVectorStore
from llm_judge import LLMJudge
from tqdm import tqdm

def run_full_pipeline(input_csv="Data/test.csv", output_csv="results.csv"):
    
    print("="*60)
    print("BACKSTORY CONSISTENCY CHECKER - FULL PIPELINE")
    print("="*60)
    
    print("\n1. Loading vector store...")
    store = NovelVectorStore()
    if Path("Data/vector_store.pkl").exists():
        store.load()
    else:
        print("Building vector store from scratch...")
        store.build_from_novels()
        store.save()
    
    print("\n2. Initializing LLM judge...")
    judge = LLMJudge(store, model_name="qwen2.5:3b")
    
    print(f"\n3. Loading test data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} examples to process")
    
    book_map = {
        'In Search of the Castaways': 'castaways',
        'The Count of Monte Cristo': 'monte'
    }
    
    print("\n4. Processing examples...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            book_name = book_map.get(row['book_name'], 'monte')
            caption = row.get('caption', '')
            backstory = row['content']
            
            result = judge.judge_consistency(
                backstory=backstory,
                book_name=book_name,
                caption=str(caption) if pd.notna(caption) else ""
            )
            
            results.append({
                'Story ID': row['id'],
                'Prediction': result['prediction'],
                'Rationale': result['reasoning']
            })
            
        except Exception as e:
            print(f"\nError processing ID {row['id']}: {e}")
            results.append({
                'Story ID': row['id'],
                'Prediction': 1,
                'Rationale': f"Error: {str(e)[:100]}"
            })
    
    print(f"\n5. Saving results to {output_csv}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Done! Processed {len(results)} examples")
    print(f"Results saved to: {output_csv}")
    
    print("\nPrediction distribution:")
    print(results_df['Prediction'].value_counts())
    
    return results_df


def validate_on_train():
    """Validate accuracy on training data"""
    print("="*60)
    print("VALIDATION ON TRAINING DATA")
    print("="*60)
    
    print("\nLoading vector store...")
    store = NovelVectorStore()
    store.load()
    
    print("Initializing LLM judge...")
    judge = LLMJudge(store, model_name="qwen2.5:3b")
    
    print("Loading training data...")
    train = pd.read_csv("Data/train-2.csv")
    
    sample = train.sample(n=10, random_state=42)
    
    book_map = {
        'In Search of the Castaways': 'castaways',
        'The Count of Monte Cristo': 'monte'
    }
    
    correct = 0
    total = 0
    
    print(f"\nTesting on {len(sample)} examples...\n")
    
    for idx, row in sample.iterrows():
        book_name = book_map.get(row['book_name'], 'monte')
        
        result = judge.judge_consistency(
            backstory=row['content'],
            book_name=book_name
        )
        
        true_label = 1 if row['label'] == 'consistent' else 0
        predicted = result['prediction']
        
        is_correct = (predicted == true_label)
        correct += is_correct
        total += 1
        
        status = "correct" if is_correct else "wrong"
        print(f"{status} ID {row['id']}: Predicted {predicted}, Actual {true_label}")
    
    accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"Validation Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_on_train()
    else:
        run_full_pipeline(
            input_csv="Data/test.csv",
            output_csv="results.csv"
        )