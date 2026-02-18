"""
File I/O operations for TSV parsing and JSON export.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def load_articles(filepath: str) -> pd.DataFrame:
    """
    Load articles from a tab-delimited file.
    
    Args:
        filepath: Path to TSV file
        
    Returns:
        DataFrame with article_id and title columns
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    try:
        # Read TSV file
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
        
        # Validate columns
        required_columns = ['article_id', 'title']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Found columns: {df.columns.tolist()}"
            )
        
        # Clean data
        df = df.dropna(subset=['title'])  # Remove rows with empty titles
        df = df[df['title'].str.strip() != '']  # Remove blank titles
        
        # Ensure article_id is string
        df['article_id'] = df['article_id'].astype(str)
        
        logger.info(f"Loaded {len(df)} valid articles from {filepath}")
        
        if df.empty:
            logger.warning("No valid articles found after cleaning")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        return pd.DataFrame(columns=['article_id', 'title'])
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise


def export_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Export categorization results to JSON file.
    
    Args:
        results: Dictionary containing categorization results
        output_path: Path to output JSON file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        export_data = {
            'categories': list(results['categories'].values()),
            'uncategorized': results['uncategorized']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        raise IOError(f"Could not write to {output_path}: {e}")
