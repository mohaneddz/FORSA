# picks 12 random chunks from the files (/gen/<files>.md)

import random
import os

def pick_random_chunks(chunks: list[str], n: int = 12) -> list[str]:
    """Pick n random chunks from the list of chunks."""
    if len(chunks) <= n:
        return chunks
    return random.sample(chunks, n)

def load_chunks_from_gen(gen_dir: str = "gen") -> list[dict]:
    """Load all markdown chunks from the gen directory."""
    chunks = []
    
    if not os.path.exists(gen_dir):
        print(f"Directory '{gen_dir}' not found!")
        return chunks
    
    # Get all .md files from gen directory
    md_files = [f for f in os.listdir(gen_dir) if f.endswith('.md')]
    
    for md_file in md_files:
        file_path = os.path.join(gen_dir, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks.append({
                    'filename': md_file,
                    'content': content
                })
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    
    return chunks

# Example usage
if __name__ == "__main__":
    # Load all chunks from gen directory
    all_chunks = load_chunks_from_gen()
    print(f"Loaded {len(all_chunks)} chunks from gen/ directory\n")
    
    # Pick 12 random chunks
    selected_chunks = pick_random_chunks(all_chunks, n=12)
    
    # Print selected chunks
    print(f"Selected {len(selected_chunks)} random chunks:\n")
    print("=" * 80)
    
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"\n### Chunk {i}: {chunk['filename']}")
        print("-" * 80)
        print(chunk['content'])
        print("=" * 80)
    
    # Save to file
    output_content = ""
    for i, chunk in enumerate(selected_chunks, 1):
        output_content += f"\n### Chunk {i}: {chunk['filename']}\n"
        output_content += "-" * 80 + "\n"
        output_content += chunk['content'] + "\n"
        output_content += "=" * 80 + "\n"
    
    with open("random.md", "w", encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✓ Saved {len(selected_chunks)} random chunks to random.md")