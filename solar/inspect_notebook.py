import json

notebook_path = "e:/solar/Regression-solar power generation(EDA) (1).ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Found {len(nb.get('cells', []))} cells.")
    
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "train_test_split" in source or "model" in source or ".fit" in source or "X =" in source or "X=" in source:
                print(f"--- Cell {i} ---")
                print(source)
                print("----------------\n")

except Exception as e:
    print(f"Error: {e}")
