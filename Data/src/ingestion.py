import pathway as pw
import pandas as pd
from pathlib import Path

def break_into_chunks(text,chunk_size=800):
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        chunk = " ".join(chunk)
        yield chunk
    
def load_books():
    books_dir = Path("Data/Novels")
    castaways_path= books_dir / "castaways.txt"
    monte_path= books_dir / "Monte_Cristo.txt"

    castaways_text = castaways_path.read_text(encoding="utf-8", errors="ignore")
    monte_text = monte_path.read_text(encoding="utf-8", errors="ignore")


    return {
        "monte" : monte_text,
        "castaways" : castaways_text
    } 

def build_novel_rows():
    rows=[]
    books = load_books()
    for name, text in books.items():
        chunks = break_into_chunks(text)
        for idx, chunk in enumerate(chunks):
            row = (name, idx, chunk)
            rows.append(row)
    return rows

def build_novel_table():
    rows = build_novel_rows()
    table = pw.debug.table_from_rows(
        schema=pw.schema_builder({
            "book": pw.column_definition(dtype=str),
            "chunk_id": pw.column_definition(dtype=int),
            "content": pw.column_definition(dtype=str)
        }),
        rows=rows
    )
    return table

def load_train():
    path= ("Data/train-2.csv")
    df=pd.read_csv("Data/train-2.csv")
    return df

if __name__ == "__main__":
    train = load_train()
    print("TRAIN SHAPE:", train.shape)
    print(train.head(5))

