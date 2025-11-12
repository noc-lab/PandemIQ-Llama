import os
import json


def load_text_file(file_path: str) -> str:
    """Load prompt text from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
def load_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def save2jsonl(results, output_file):
    """
    Append one or more results to a JSONL file.
    Each element in `results` will be written on its own line.
    """
    with open(output_file, "a", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
def write2jsonl(results, output_file):
    """
    Append one or more results to a JSONL file.
    Each element in `results` will be written on its own line.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]