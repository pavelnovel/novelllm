from pathlib import Path
import re
import json

CORPUS_DIR = Path("corpus")
CHUNKS_ROOT = Path("chunks")
SEPARATOR = "---"  # change here if you use a different delimiter

# Process both .md and .json files
for file_path in CORPUS_DIR.glob("*.[mj][ds]*"):  # matches both .md and .json
    # Create a more informative directory name
    out_dir = CHUNKS_ROOT / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Read file content
    raw_text = file_path.read_text(encoding="utf-8")
    
    if file_path.suffix == '.json':
        try:
            # Parse JSON and convert to string representation
            json_data = json.loads(raw_text)
            chunks = [json.dumps(item, ensure_ascii=False) for item in json_data]
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {file_path.name}")
            continue
    else:  # .md files
        # Use regex to split on lines that contain only the separator
        chunks = re.split(f"^{SEPARATOR}$", raw_text, flags=re.MULTILINE)
    
    # Process and save chunks
    chunk_count = 0
    for i, chunk in enumerate(chunks):
        cleaned = chunk.strip()
        if cleaned:
            (out_dir / f"chunk_{chunk_count:03}.txt").write_text(cleaned, encoding="utf-8")
            chunk_count += 1
    
    print(f"Processed {file_path.name}: Created {chunk_count} chunks in {out_dir}")