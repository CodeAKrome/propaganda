#!/usr/bin/env python3
import sys
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Backend availability checks
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Gemini configuration
try:
    import google.generativeai as genai
    gemini_configured = False
    if os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        gemini_configured = True
except ImportError:
    gemini_configured = False


class NewsCategorizer:
    """Categorizes news article titles into a three-level hierarchy using LLMs."""
    
    def __init__(self, model_name: str, backend: str = "mlx", max_batch_size: int = 100):
        """
        Initialize the categorizer with a model and backend.
        
        Args:
            model_name: Name/path of the model to use
            backend: Backend type: 'mlx', 'ollama', or 'gemini'
            max_batch_size: Maximum titles to process per LLM call
        """
        self.model_name = model_name
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        if backend == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama package not installed. Install with: pip install ollama")
            print(f"Using Ollama model: {model_name}", file=sys.stderr)
        elif backend == "gemini":
            if not gemini_configured:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            print(f"Using Gemini model: {model_name}", file=sys.stderr)
            self.model = genai.GenerativeModel(model_name)
        else:  # mlx
            if not MLX_AVAILABLE:
                raise ImportError("mlx_lm package not installed. Install with: pip install mlx-lm")
            print(f"Loading MLX model: {model_name}...", file=sys.stderr)
            self.model, self.tokenizer = load(model_name)
    
    def parse_titles_file(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Parse titles file into list of (id, title) tuples.
        
        Expected format: ID<tab>Title (with optional surrounding quotes)
        """
        titles = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split on first tab
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} malformed - no tab separator", file=sys.stderr)
                    continue
                
                id_part, title_part = parts
                id_part = id_part.strip()
                title_part = title_part.strip()
                
                # Remove surrounding quotes if present
                if len(title_part) >= 2 and (
                    (title_part[0] == "'" and title_part[-1] == "'") or
                    (title_part[0] == '"' and title_part[-1] == '"')
                ):
                    title_part = title_part[1:-1]
                
                if id_part and title_part:
                    titles.append((id_part, title_part))
        
        print(f"Successfully parsed {len(titles)} titles", file=sys.stderr)
        return titles
    
    def create_categorization_prompt(self, batch: List[Tuple[str, str]]) -> str:
        """
        Create a structured prompt for the LLM to categorize titles.
        
        Args:
            batch: List of (id, title) tuples to categorize
            
        Returns:
            Formatted prompt string
        """
        titles_json = json.dumps(
            [{"id": id_, "title": title} for id_, title in batch],
            indent=2
        )
        
        prompt = f"""You are a precise news categorization system. Categorize each article into exactly three hierarchical levels.

CATEGORIZATION RULES:
- Level 1: Broad domain (e.g., World, US Politics, Business/Economy, Science/Tech, Sports, Entertainment, Health, Environment, Law/Justice, Culture)
- Level 2: Geographic region or sector focus (e.g., Europe, Asia, Middle East, US Economy, Corporate, Energy, Technology)
- Level 3: Specific topic/event (e.g., Russia-Ukraine War, Elections, Immigration, Climate Change, Trade, Courts)

CRITICAL REQUIREMENTS:
1. Use EXACTLY three levels for every article
2. Be consistent in naming (e.g., always "Russia-Ukraine War" not "Ukraine Crisis")
3. Keep the original ID and title unchanged
4. Return ONLY valid JSON array with no additional text

OUTPUT FORMAT:
[
  {{
    "id": "original_id",
    "title": "Original Title",
    "level1": "Broad Category",
    "level2": "Regional/Sector Focus",
    "level3": "Specific Topic"
  }}
]

Process these {len(batch)} articles:

{titles_json}"""
        return prompt
    
    def categorize_batch(self, batch: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a single batch of titles through the LLM.
        
        Args:
            batch: List of (id, title) tuples
            
        Returns:
            List of categorized articles as dicts
        """
        prompt = self.create_categorization_prompt(batch)
        
        # Generate response based on backend
        try:
            if self.backend == "ollama":
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response["message"]["content"]
            elif self.backend == "gemini":
                response = self.model.generate_content(prompt)
                result_text = response.text
            else:  # mlx
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                result_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    verbose=False,
                    max_tokens=4096
                )
        except Exception as e:
            print(f"Error generating response: {e}", file=sys.stderr)
            return []
        
        # Robust JSON extraction
        try:
            # Find JSON array boundaries
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = result_text[start:end]
                categorized = json.loads(json_str)
                
                # Validate result count matches input count
                if len(categorized) != len(batch):
                    print(
                        f"Warning: Batch size mismatch: expected {len(batch)}, got {len(categorized)}",
                        file=sys.stderr
                    )
                
                return categorized
            else:
                raise ValueError("No JSON array found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}", file=sys.stderr)
            print(f"Raw response: {result_text[:500]}...", file=sys.stderr)
            return []
    
    def categorize_all(self, titles: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Process all titles in batches and combine results.
        
        Args:
            titles: Complete list of (id, title) tuples
            
        Returns:
            Combined list of all categorized articles
        """
        all_results = []
        total_batches = (len(titles) + self.max_batch_size - 1) // self.max_batch_size
        
        for i in range(0, len(titles), self.max_batch_size):
            batch_num = i // self.max_batch_size + 1
            batch = titles[i:i + self.max_batch_size]
            
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} titles)...",
                file=sys.stderr
            )
            
            batch_results = self.categorize_batch(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def sort_and_format(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Sort by category hierarchy and format output lines.
        
        Args:
            results: List of categorized article dicts
            
        Returns:
            Formatted tab-delimited output lines
        """
        # Ensure all required fields exist
        for result in results:
            result['level1'] = result.get('level1', 'Uncategorized')
            result['level2'] = result.get('level2', 'Uncategorized')
            result['level3'] = result.get('level3', 'Uncategorized')
        
        # Sort by hierarchy and title
        sorted_results = sorted(results, key=lambda x: (
            x['level1'].lower(),
            x['level2'].lower(),
            x['level3'].lower(),
            x['title'].lower()
        ))
        
        # Format as tab-delimited lines
        return [
            f"{r['level1']}\t{r['level2']}\t{r['level3']}\t{r['id']}\t{r['title']}"
            for r in sorted_results
        ]


def main():
    """Main entry point for the categorization tool."""
    parser = argparse.ArgumentParser(
        description="Categorize news article titles using LLM with 3-level hierarchy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use MLX (default)
  python titlesort.py titles.txt --model mlx-community/Llama-3.3-70B-Instruct-8bit --output sorted_news.tsv
  
  # Use Ollama
  python titlesort.py titles.txt --ollama --model llama3.3 --output sorted_news.tsv
  
  # Use Gemini
  export GEMINI_API_KEY="your-api-key"
  python titlesort.py titles.txt --gemini --model models/gemini-1.5-flash --output sorted_news.tsv"""
    )
    
    parser.add_argument(
        "titles_file",
        help="Path to input file with tab-separated ID and article titles"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.3-70B-Instruct-8bit",
        help="Model name (default: mlx-community/Llama-3.3-70B-Instruct-8bit)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Max titles per LLM batch (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output TSV file path (default: stdout)"
    )
    
    # Backend selection (mutually exclusive)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama instead of MLX for inference"
    )
    backend_group.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini instead of MLX for inference"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.titles_file).exists():
        print(f"Error: File '{args.titles_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Determine backend
    backend = "mlx"
    if args.ollama:
        backend = "ollama"
    elif args.gemini:
        backend = "gemini"
    
    # Validate backend availability
    if backend == "ollama" and not OLLAMA_AVAILABLE:
        print(f"Error: ollama package not installed. Install with: pip install ollama", file=sys.stderr)
        sys.exit(1)
    elif backend == "gemini" and not gemini_configured:
        print(f"Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    elif backend == "mlx" and not MLX_AVAILABLE:
        print(f"Error: mlx_lm package not installed. Install with: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print("=" * 70, file=sys.stderr)
    print(f"Backend: {backend.capitalize()}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Batch size: {args.batch_size}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    # Initialize system
    try:
        categorizer = NewsCategorizer(args.model, backend, args.batch_size)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse input
    print("Reading input file...", file=sys.stderr)
    titles = categorizer.parse_titles_file(args.titles_file)
    
    if not titles:
        print("Error: No valid titles found in file", file=sys.stderr)
        sys.exit(1)
    
    # Categorize
    print(f"Categorizing {len(titles)} articles...", file=sys.stderr)
    results = categorizer.categorize_all(titles)
    
    if not results:
        print("Error: No results generated", file=sys.stderr)
        sys.exit(1)
    
    # Sort and format
    print("Sorting results...", file=sys.stderr)
    output_lines = categorizer.sort_and_format(results)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("Level1\tLevel2\tLevel3\tID\tTitle\n")  # Header
            for line in output_lines:
                f.write(line + '\n')
        print(f"Done! Wrote {len(output_lines)} categorized articles to {args.output}", file=sys.stderr)
    else:
        # Write to stdout with header
        print("Level1\tLevel2\tLevel3\tID\tTitle")
        for line in output_lines:
            print(line)


if __name__ == "__main__":
    main()