import argparse
import re
import unicodedata
import tiktoken

# 1. Normalization function (from normalize.py)
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]

def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for z in ZERO_WIDTH:
        text = text.replace(z, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # Setup CLI arguments using argparse (Requirement 1)
    parser = argparse.ArgumentParser(description="A small CLI to analyze text files using tiktoken.")
    parser.add_argument("filepath", type=str, help="Path to the input text file")
    parser.add_argument("--encoding", type=str, default="cl100k_base", help="Tiktoken encoding name (default: cl100k_base)")
    parser.add_argument("--viz", action="store_true", help="Visualize token pieces in brackets (Stretch goal)")
    
    args = parser.parse_args()

    # Read the file content
    try:
        with open(args.filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.filepath}' not found.")
        return

    # Initialize tokenizer
    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        print(f"Error loading encoding '{args.encoding}': {e}")
        return

    # Token count before normalization
    raw_ids = enc.encode(raw_text)
    
    # Apply canonical normalization (Requirement 4)
    clean_text = canonical(raw_text)
    clean_ids = enc.encode(clean_text)

    # Calculate metrics
    chars = len(clean_text)
    bytes_data = clean_text.encode("utf-8")
    num_bytes = len(bytes_data)
    num_tokens = len(clean_ids)
    bpt = num_bytes / num_tokens if num_tokens > 0 else 0

    # Pricing estimation (Requirement 5: e.g., $0.15 per 1M tokens)
    price_per_token = 0.15 / 1_000_000
    estimated_cost = num_tokens * price_per_token

    # Print the report (Requirement 2)
    print("=" * 50)
    print(f"TOKEN REPORT: {args.filepath}")
    print(f"Encoding Model: {args.encoding}")
    print("=" * 50)
    print(f"Original Chars           : {len(raw_text)}")
    print(f"Normalized Chars         : {chars}")
    print(f"Bytes                    : {num_bytes}")
    print(f"Tokens (Before Clean)    : {len(raw_ids)}")
    print(f"Tokens (After Clean)     : {num_tokens}")
    print(f"Bytes / Token            : {bpt:.2f}")
    print(f"Estimated Input Cost     : ${estimated_cost:.6f}")
    print("-" * 50)

    # Stretch goal: --viz flag (Show token pieces)
    if args.viz:
        print("Token Visualization (--viz):")
        out = []
        for tid in clean_ids:
            piece = enc.decode([tid]).replace(" ", ".")
            out.append(f"[{piece}]")
        print("".join(out))
        print("-" * 50)

if __name__ == "__main__":
    main()