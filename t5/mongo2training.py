"""
MongoDB to Bias Training Data Exporter
=======================================
Exports article and bias data from MongoDB to JSON format for training
the political bias detector model.

FIXED: Handles bias field stored as JSON string or object.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure


# ==============================================================================
# MONGODB CONNECTION
# ==============================================================================

class MongoDBConnection:
    """Handles MongoDB connection and operations."""
    
    def __init__(
        self,
        connection_string: str = None,
        database: str = "rssnews",
        collection: str = "articles"
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string (URI)
            database: Database name
            collection: Collection name
        """
        # Use connection string from arg, env var, or default to localhost
        self.connection_string = (
            connection_string or
            os.getenv('MONGODB_URI') or
            'mongodb://localhost:27017/'
        )
        
        self.database_name = database
        self.collection_name = collection
        
        try:
            print(f"Connecting to MongoDB...")
            self.client = MongoClient(self.connection_string)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            print(f"✓ Connected to database: {self.database_name}")
            print(f"✓ Using collection: {self.collection_name}")
            
        except ConnectionFailure as e:
            print(f"Error: Failed to connect to MongoDB: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Unexpected error during connection: {e}")
            sys.exit(1)
    
    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            print("MongoDB connection closed")


# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

class BiasDataExtractor:
    """Extracts and formats bias training data from MongoDB."""
    
    def __init__(self, mongo_conn: MongoDBConnection):
        """
        Initialize extractor.
        
        Args:
            mongo_conn: MongoDBConnection instance
        """
        self.mongo = mongo_conn
        self.collection = mongo_conn.collection
    
    def build_query(
        self,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build MongoDB query based on filters.
        
        FIXED: Handles bias as both string and object types.
        
        Args:
            source: Filter by news source
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            skip: Number of records to skip
            
        Returns:
            Dictionary with 'query' and 'options' keys
        """
        query = {}
        
        # CRITICAL FIX: Handle bias field as string or object
        # Check that bias field exists and is not null/empty
        # The bias field might be stored as a JSON string OR as an object
        query['bias'] = {
            '$exists': True,
            '$ne': None,
            '$ne': '',  # Also exclude empty strings
            '$type': ['string', 'object']  # Accept either type
        }
        
        # Filter by source if provided
        if source:
            query['source'] = source
        
        # Filter by date range if provided
        if start_date or end_date:
            date_query = {}
            
            if start_date:
                try:
                    start = datetime.strptime(start_date, '%Y-%m-%d')
                    date_query['$gte'] = start
                except ValueError:
                    print(f"Warning: Invalid start_date format: {start_date}")
                    print("Expected format: YYYY-MM-DD")
            
            if end_date:
                try:
                    end = datetime.strptime(end_date, '%Y-%m-%d')
                    # Include the entire end date
                    end = end.replace(hour=23, minute=59, second=59)
                    date_query['$lte'] = end
                except ValueError:
                    print(f"Warning: Invalid end_date format: {end_date}")
                    print("Expected format: YYYY-MM-DD")
            
            if date_query:
                # Try multiple possible date field names
                query['$or'] = [
                    {'date': date_query},
                    {'published_date': date_query},
                    {'published': date_query},
                    {'created_at': date_query}
                ]
        
        # Build options
        options = {}
        if limit:
            options['limit'] = limit
        if skip:
            options['skip'] = skip
        
        print(f"q: {query} ops: {options}")

        return {'query': query, 'options': options}
    
    def extract_training_data(
        self,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Extract training data from MongoDB.
        
        Args:
            source: Filter by news source
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            skip: Number of records to skip
            verbose: Print detailed information
            
        Returns:
            List of training data dictionaries
        """
        # Build query
        query_info = self.build_query(
            source=source,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            skip=skip
        )
        
        query = query_info['query']
        options = query_info['options']
        
        if verbose:
            print(f"\nQuery: {json.dumps(query, indent=2, default=str)}")
            print(f"Options: {json.dumps(options, indent=2)}")
        
        # Count matching documents
        try:
            total_count = self.collection.count_documents(query)
            print(f"\nFound {total_count} documents matching query")
            
            if total_count == 0:
                print("No documents found with the specified filters")
                print("\nDEBUG: Checking total documents and bias field...")
                total_all = self.collection.count_documents({})
                print(f"  Total documents in collection: {total_all}")
                
                # Check different bias field conditions
                bias_exists = self.collection.count_documents({'bias': {'$exists': True}})
                bias_not_null = self.collection.count_documents({'bias': {'$ne': None}})
                bias_not_empty = self.collection.count_documents({'bias': {'$ne': ''}})
                bias_string = self.collection.count_documents({'bias': {'$type': 'string'}})
                bias_object = self.collection.count_documents({'bias': {'$type': 'object'}})
                
                print(f"  Documents with 'bias' field existing: {bias_exists}")
                print(f"  Documents with 'bias' not null: {bias_not_null}")
                print(f"  Documents with 'bias' not empty string: {bias_not_empty}")
                print(f"  Documents with 'bias' as string: {bias_string}")
                print(f"  Documents with 'bias' as object: {bias_object}")
                
                # Show a sample document
                sample = self.collection.find_one({'bias': {'$exists': True}})
                if sample:
                    print(f"\nSample document structure:")
                    print(f"  _id: {sample.get('_id')}")
                    print(f"  bias type: {type(sample.get('bias'))}")
                    print(f"  bias value (first 100 chars): {str(sample.get('bias'))[:100]}")
                
                return []
            
        except OperationFailure as e:
            print(f"Error counting documents: {e}")
            return []
        
        # Fetch documents
        print(f"Extracting data...")
        
        try:
            cursor = self.collection.find(query, **options)
            
            training_data = []
            processed = 0
            skipped = 0
            
            for doc in cursor:
                processed += 1
                
                # Extract article text
                article_text = doc.get('article', '')
                
                if not article_text or not isinstance(article_text, str):
                    if verbose:
                        print(f"Warning: Skipping document {doc.get('_id')} - missing/invalid article field")
                    skipped += 1
                    continue
                
                # Extract bias data
                # CRITICAL FIX: Handle bias as string or object
                bias_raw = doc.get('bias')
                
                if not bias_raw:
                    if verbose:
                        print(f"Warning: Skipping document {doc.get('_id')} - missing bias field")
                    skipped += 1
                    continue
                
                # Parse bias data if it's a string
                if isinstance(bias_raw, str):
                    try:
                        bias_data = json.loads(bias_raw)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"Warning: Skipping document {doc.get('_id')} - invalid JSON in bias field: {e}")
                        skipped += 1
                        continue
                elif isinstance(bias_raw, dict):
                    bias_data = bias_raw
                else:
                    if verbose:
                        print(f"Warning: Skipping document {doc.get('_id')} - bias field is neither string nor object")
                    skipped += 1
                    continue
                
                # Validate bias structure
                if not self._validate_bias_structure(bias_data, doc.get('_id'), verbose):
                    skipped += 1
                    continue
                
                # Create training entry
                entry = {
                    'article': article_text.strip(),
                    'label': bias_data
                }
                
                # Optionally include metadata (commented out by default)
                # Uncomment if you want to preserve metadata
                # entry['metadata'] = {
                #     '_id': str(doc.get('_id')),
                #     'source': doc.get('source'),
                #     'title': doc.get('title'),
                #     'date': doc.get('date', doc.get('published_date', doc.get('published'))),
                # }
                
                training_data.append(entry)
                
                # Progress indicator
                if verbose and processed % 100 == 0:
                    print(f"Processed {processed} documents...")
            
            print(f"\n✓ Successfully extracted {len(training_data)} training samples")
            if skipped > 0:
                print(f"⚠ Skipped {skipped} documents due to missing/invalid data")
            
            return training_data
            
        except OperationFailure as e:
            print(f"Error querying documents: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during extraction: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _validate_bias_structure(
        self,
        bias_data: Dict,
        doc_id: Any,
        verbose: bool = False
    ) -> bool:
        """
        Validate bias data structure.
        
        Expected structure:
        {
            "dir": {"L": float, "C": float, "R": float},
            "deg": {"L": float, "M": float, "H": float},
            "reason": str
        }
        
        Args:
            bias_data: Bias dictionary to validate
            doc_id: Document ID for error messages
            verbose: Print validation errors
            
        Returns:
            True if valid, False otherwise
        """
        # Check top-level keys
        required_keys = ['dir', 'deg', 'reason']
        for key in required_keys:
            if key not in bias_data:
                if verbose:
                    print(f"Warning: Document {doc_id} - missing '{key}' in bias data")
                return False
        
        # Validate 'dir' structure
        if not isinstance(bias_data['dir'], dict):
            if verbose:
                print(f"Warning: Document {doc_id} - 'dir' must be a dictionary")
            return False
        
        dir_keys = ['L', 'C', 'R']
        for key in dir_keys:
            if key not in bias_data['dir']:
                if verbose:
                    print(f"Warning: Document {doc_id} - missing 'dir.{key}'")
                return False
            if not isinstance(bias_data['dir'][key], (int, float)):
                if verbose:
                    print(f"Warning: Document {doc_id} - 'dir.{key}' must be a number")
                return False
        
        # Validate 'deg' structure
        if not isinstance(bias_data['deg'], dict):
            if verbose:
                print(f"Warning: Document {doc_id} - 'deg' must be a dictionary")
            return False
        
        deg_keys = ['L', 'M', 'H']
        for key in deg_keys:
            if key not in bias_data['deg']:
                if verbose:
                    print(f"Warning: Document {doc_id} - missing 'deg.{key}'")
                return False
            if not isinstance(bias_data['deg'][key], (int, float)):
                if verbose:
                    print(f"Warning: Document {doc_id} - 'deg.{key}' must be a number")
                return False
        
        # Validate 'reason'
        if not isinstance(bias_data['reason'], str):
            if verbose:
                print(f"Warning: Document {doc_id} - 'reason' must be a string")
            return False
        
        if not bias_data['reason'].strip():
            if verbose:
                print(f"Warning: Document {doc_id} - 'reason' cannot be empty")
            return False
        
        return True
    
    def get_statistics(
        self,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about available data.
        
        Args:
            source: Filter by news source
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with statistics
        """
        query_info = self.build_query(
            source=source,
            start_date=start_date,
            end_date=end_date
        )
        
        query = query_info['query']
        
        stats = {
            'total_with_bias': self.collection.count_documents(query),
            'total_all': self.collection.count_documents({}),
        }
        
        # Get unique sources
        try:
            sources = self.collection.distinct('source', query)
            stats['sources'] = sources
            stats['source_count'] = len(sources)
        except:
            stats['sources'] = []
            stats['source_count'] = 0
        
        # Additional stats
        stats['bias_as_string'] = self.collection.count_documents({
            'bias': {'$type': 'string', '$ne': ''}
        })
        stats['bias_as_object'] = self.collection.count_documents({
            'bias': {'$type': 'object'}
        })
        
        return stats


# ==============================================================================
# FILE EXPORT
# ==============================================================================

class BiasDataExporter:
    """Exports training data to JSON file."""
    
    @staticmethod
    def export_to_json(
        data: List[Dict],
        output_file: str,
        pretty: bool = True,
        include_metadata: bool = False
    ):
        """
        Export training data to JSON file.
        
        Args:
            data: List of training data dictionaries
            output_file: Output file path
            pretty: Use pretty printing (indented)
            include_metadata: Include metadata in export
        """
        if not data:
            print("No data to export")
            return
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = []
        for entry in data:
            export_entry = {
                'article': entry['article'],
                'label': entry['label']
            }
            
            if include_metadata and 'metadata' in entry:
                export_entry['metadata'] = entry['metadata']
            
            export_data.append(export_entry)
        
        # Write to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(export_data, f, ensure_ascii=False)
            
            print(f"\n✓ Exported {len(export_data)} samples to: {output_file}")
            
            # Show file size
            file_size = output_path.stat().st_size
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            
            print(f"  File size: {size_str}")
            
        except IOError as e:
            print(f"Error writing to file: {e}")
        except Exception as e:
            print(f"Unexpected error during export: {e}")
    
    @staticmethod
    def preview_data(data: List[Dict], num_samples: int = 3):
        """
        Print preview of exported data.
        
        Args:
            data: List of training data dictionaries
            num_samples: Number of samples to preview
        """
        if not data:
            print("No data to preview")
            return
        
        print(f"\n{'='*80}")
        print(f"DATA PREVIEW (showing {min(num_samples, len(data))} of {len(data)} samples)")
        print(f"{'='*80}\n")
        
        for i, entry in enumerate(data[:num_samples], 1):
            print(f"Sample {i}:")
            print(f"  Article: {entry['article'][:150]}...")
            print(f"  Label:")
            print(f"    Direction: {entry['label']['dir']}")
            print(f"    Degree: {entry['label']['deg']}")
            print(f"    Reason: {entry['label']['reason'][:100]}...")
            print()


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export bias training data from MongoDB to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all records with bias data
  python mongo2training.py -o training_data.json
  
  # Export from specific source
  python mongo2training.py -o data.json --source "fox-latest"
  
  # Export with date range
  python mongo2training.py -o data.json --start-date 2024-01-01 --end-date 2024-12-31
  
  # Limit number of records
  python mongo2training.py -o data.json --limit 100
  
  # Show statistics only
  python mongo2training.py --stats
  
  # Use custom MongoDB connection
  python mongo2training.py -o data.json --uri "mongodb://user:pass@host:27017/"
  
  # Verbose output with preview
  python mongo2training.py -o data.json --verbose --preview 5
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output JSON file path (required unless using --stats)'
    )
    
    # MongoDB connection
    parser.add_argument(
        '--uri',
        type=str,
        help='MongoDB connection string (default: localhost or MONGODB_URI env var)'
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default='rssnews',
        help='Database name (default: news_db)'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default='articles',
        help='Collection name (default: articles)'
    )
    
    # Query filters (same as mongo2chroma.py)
    parser.add_argument(
        '--source',
        type=str,
        help='Filter by news source'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date filter (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date filter (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of records to export'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        help='Number of records to skip'
    )
    
    # Output options
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use compact JSON format (no indentation)'
    )
    
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include document metadata in export'
    )
    
    # Information options
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit (no export)'
    )
    
    parser.add_argument(
        '--preview',
        type=int,
        metavar='N',
        help='Preview N samples before exporting'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    return parser.parse_args()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.stats and not args.output:
        print("Error: --output is required (unless using --stats)")
        sys.exit(1)
    
    print("="*80)
    print("MongoDB to Bias Training Data Exporter")
    print("="*80)
    
    # Connect to MongoDB
    mongo_conn = MongoDBConnection(
        connection_string=args.uri,
        database=args.database,
        collection=args.collection
    )
    
    try:
        extractor = BiasDataExtractor(mongo_conn)
        
        # Show statistics if requested
        if args.stats:
            print("\nGathering statistics...")
            stats = extractor.get_statistics(
                source=args.source,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            print(f"\n{'='*80}")
            print("STATISTICS")
            print(f"{'='*80}")
            print(f"Total documents in collection: {stats['total_all']}")
            print(f"Documents with bias data: {stats['total_with_bias']}")
            if stats['total_all'] > 0:
                print(f"Coverage: {stats['total_with_bias']/stats['total_all']*100:.1f}%")
            print(f"\nBias field storage:")
            print(f"  As JSON string: {stats['bias_as_string']}")
            print(f"  As object: {stats['bias_as_object']}")
            print(f"\nUnique sources: {stats['source_count']}")
            if stats['sources']:
                for source in sorted(stats['sources']):
                    print(f"  - {source}")
            
            return
        
        # Extract data
        print("\nExtracting training data from MongoDB...")
        training_data = extractor.extract_training_data(
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            skip=args.skip,
            verbose=args.verbose
        )
        
        if not training_data:
            print("\nNo data extracted. Exiting.")
            return
        
        # Preview if requested
        if args.preview:
            BiasDataExporter.preview_data(training_data, num_samples=args.preview)
        
        # Export to file
        BiasDataExporter.export_to_json(
            data=training_data,
            output_file=args.output,
            pretty=not args.compact,
            include_metadata=args.include_metadata
        )
        
        print("\n✓ Export completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExport cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    finally:
        mongo_conn.close()


if __name__ == "__main__":
    main()
