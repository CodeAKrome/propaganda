#!/usr/bin/env python3
"""
MongoDB Regex Search CLI Tool (with env var auth)

Searches a MongoDB collection for documents where string fields match a regex.

Authentication:
- If MONGO_USER and MONGO_PASS environment variables are set → uses them.
- If not set → connects without authentication (useful for local unsecured instances).

Export variables before running:
  export MONGO_USER="myuser"
  export MONGO_PASS="mypassword"
"""

import os
import sys
from typing import List, Optional

import fire
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import re
from tqdm import tqdm


class MongoRegexSearch:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        auth_source: str = "admin",
        unicode_decode_error_handler: str = "strict",
    ):
        """
        Initialize MongoDB connection using MONGO_USER and MONGO_PASS env vars if available.
        
        Args:
            unicode_decode_error_handler: How to handle unicode decode errors.
                Options: 'strict' (default, raise errors), 'ignore' (skip bad chars), 
                'replace' (replace with �)
        """
        username = os.getenv("MONGO_USER")
        password = os.getenv("MONGO_PASS")

        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_source}"
            print("Connecting with provided MONGO_USER and MONGO_PASS credentials.")
        else:
            connection_string = f"mongodb://{host}:{port}/"
            print("No MONGO_USER/MONGO_PASS found → connecting without authentication.")

        try:
            self.client = MongoClient(
                connection_string,
                unicode_decode_error_handler=unicode_decode_error_handler
            )
            # Test the connection
            self.client.admin.command("ping")
            print("Connected to MongoDB successfully.")
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Connection error: {e}")
            sys.exit(1)

    def _is_string_field(self, value) -> bool:
        return isinstance(value, str)

    def _build_regex(self, pattern: str, ignore_case: bool = False, regex: bool = False):
        flags = re.IGNORECASE if ignore_case else 0
        if regex:
            return re.compile(pattern, flags)
        else:
            return re.compile(re.escape(pattern), flags)

    def _parse_field_list(self, fields):
        """Parse fields parameter which can be string or tuple/list from Fire."""
        if not fields:
            return None
        if isinstance(fields, str):
            return [f.strip() for f in fields.split(",")]
        else:
            # Already a tuple or list from Fire
            return [f.strip() for f in fields]

    def find_corrupted(
        self,
        database: str,
        collection: str,
        fields: Optional[str] = None,
        limit: int = 0,
        projection: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Find documents with Unicode replacement characters (�) indicating corrupted UTF-8 data.

        Args:
            database: Database name
            collection: Collection name
            fields: Comma-separated field names to check – if omitted, checks all string fields
            limit: Max results to show (0 = no limit)
            projection: Comma-separated fields to display
            show_progress: Show progress bar (default: True)
        """
        db = self.client[database]
        coll = db[collection]

        replacement_char = '\ufffd'  # Unicode replacement character �
        
        target_fields = self._parse_field_list(fields)

        # Build query if specific fields are provided
        if target_fields:
            or_conditions = [{field: {"$regex": replacement_char}} for field in target_fields]
            query = {"$or": or_conditions}
        else:
            query = {}  # We'll filter manually in Python

        try:
            cursor = coll.find(query)

            # Optional projection
            if projection:
                proj_fields = self._parse_field_list(projection)
                proj_dict = {f: 1 for f in proj_fields}
                proj_dict["_id"] = 1
                cursor = cursor.project(proj_dict)

            # For manual filtering (no target_fields), don't limit the cursor at all
            if target_fields and limit > 0:
                cursor = cursor.limit(limit)

            results = []
            
            # Create progress bar
            if target_fields:
                total_count = coll.count_documents(query)
                if limit > 0:
                    total_count = min(total_count, limit)
                pbar = tqdm(total=total_count, desc="Scanning documents", disable=not show_progress)
            else:
                pbar = tqdm(desc="Scanning documents", disable=not show_progress)
            
            for doc in cursor:
                if target_fields:
                    results.append(doc)
                    pbar.update(1)
                else:
                    # Manual check across all string fields
                    if any(
                        self._is_string_field(value) and replacement_char in value
                        for value in doc.values()
                    ):
                        results.append(doc)
                        if limit > 0 and len(results) >= limit:
                            pbar.update(1)
                            break
                    pbar.update(1)
            
            pbar.close()

            print(f"\nFound {len(results)} document(s) with corrupted UTF-8 in {database}.{collection}:\n")
            for i, doc in enumerate(results, 1):
                print(f"--- Result {i} ---")
                for k, v in doc.items():
                    if isinstance(v, str):
                        if replacement_char in v:
                            # Highlight corrupted fields
                            if len(v) > 300:
                                print(f"{k}: {v[:297]}... [CORRUPTED - contains �]")
                            else:
                                print(f"{k}: {v} [CORRUPTED - contains �]")
                        else:
                            if len(v) > 300:
                                print(f"{k}: {v[:297]}...")
                            else:
                                print(f"{k}: {v}")
                    else:
                        print(f"{k}: {v}")
                print()

        except OperationFailure as e:
            print(f"Query failed (auth/permission issue?): {e}")
        except Exception as e:
            print(f"Error during search: {e}")

    def find_utf8_errors(
        self,
        database: str,
        collection: str,
        fields: Optional[str] = None,
        limit: int = 0,
        projection: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Find documents that cause UTF-8 decode errors when read with strict validation.
        
        This command temporarily reconnects with strict UTF-8 validation to identify
        documents that have invalid UTF-8 sequences (before they get replaced with �).

        Args:
            database: Database name
            collection: Collection name
            fields: Comma-separated field names to check – if omitted, checks all string fields
            limit: Max results to show (0 = no limit)
            projection: Comma-separated fields to display
            show_progress: Show progress bar (default: True)
        """
        # Create a temporary strict client to test for UTF-8 errors
        username = os.getenv("MONGO_USER")
        password = os.getenv("MONGO_PASS")
        
        if username and password:
            connection_string = f"mongodb://{username}:{password}@localhost:27017/?authSource=admin"
        else:
            connection_string = f"mongodb://localhost:27017/"
        
        try:
            strict_client = MongoClient(
                connection_string,
                unicode_decode_error_handler='strict'
            )
            strict_db = strict_client[database]
            strict_coll = strict_db[collection]
            
            # Get IDs using the lenient client
            db = self.client[database]
            coll = db[collection]
            
            target_fields = self._parse_field_list(fields)
            
            # Get all document IDs to test
            id_cursor = coll.find({}, {"_id": 1})
            if limit > 0:
                id_cursor = id_cursor.limit(limit)
            
            doc_ids = [doc["_id"] for doc in id_cursor]
            
            results = []
            error_docs = []
            
            pbar = tqdm(total=len(doc_ids), desc="Testing documents for UTF-8 errors", disable=not show_progress)
            
            for doc_id in doc_ids:
                try:
                    # Try to read with strict validation
                    doc = strict_coll.find_one({"_id": doc_id})
                    
                    # If we get here, force iteration over all fields to trigger any errors
                    if doc:
                        if target_fields:
                            # Only check specified fields
                            for field in target_fields:
                                if field in doc and isinstance(doc[field], str):
                                    # This will trigger decode if there's invalid UTF-8
                                    _ = len(doc[field])
                        else:
                            # Check all string fields
                            for key, value in doc.items():
                                if isinstance(value, str):
                                    # This will trigger decode if there's invalid UTF-8
                                    _ = len(value)
                    pbar.update(1)
                    
                except (UnicodeDecodeError, Exception) as e:
                    # UTF-8 decode error found - fetch full doc with lenient client
                    try:
                        full_doc = coll.find_one({"_id": doc_id})
                        error_docs.append({
                            "_id": doc_id,
                            "error": str(e),
                            "doc": full_doc
                        })
                    except Exception as fetch_error:
                        error_docs.append({
                            "_id": doc_id,
                            "error": f"{str(e)} (Could not fetch full doc: {str(fetch_error)})",
                            "doc": {"_id": doc_id}
                        })
                    pbar.update(1)
            
            pbar.close()
            strict_client.close()
            
            print(f"\nFound {len(error_docs)} document(s) with UTF-8 errors in {database}.{collection}:\n")
            for i, item in enumerate(error_docs, 1):
                doc = item["doc"]
                print(f"--- Result {i} ---")
                print(f"Error: {item['error']}")
                
                if projection:
                    proj_fields = self._parse_field_list(projection)
                    doc = {k: v for k, v in doc.items() if k in proj_fields or k == "_id"}
                
                for k, v in doc.items():
                    if isinstance(v, str) and len(v) > 300:
                        print(f"{k}: {v[:297]}...")
                    else:
                        print(f"{k}: {v}")
                print()

        except Exception as e:
            print(f"Error during UTF-8 validation scan: {e}")

    def search(
        self,
        database: str,
        collection: str,
        pattern: str,
        fields: Optional[str] = None,
        ignore_case: bool = True,
        regex: bool = False,
        limit: int = 50,
        projection: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Search documents matching a regex pattern in specified or all string fields.

        Args:
            database: Database name
            collection: Collection name
            pattern: Search pattern
            fields: Comma-separated field names (e.g. "artist,album_artist") – if omitted, searches all string fields
            ignore_case: Case-insensitive (default: True)
            regex: Treat pattern as raw regex instead of escaping special chars
            limit: Max results to show (0 = no limit)
            projection: Comma-separated fields to display
            show_progress: Show progress bar (default: True)
        """
        db = self.client[database]
        coll = db[collection]

        regex_pattern = self._build_regex(pattern, ignore_case=ignore_case, regex=regex)
        target_fields = self._parse_field_list(fields)

        # Build query if specific fields are provided
        if target_fields:
            or_conditions = [{field: {"$regex": regex_pattern}} for field in target_fields]
            query = {"$or": or_conditions}
        else:
            query = {}  # We'll filter manually in Python

        try:
            cursor = coll.find(query)

            # Optional projection
            if projection:
                proj_fields = self._parse_field_list(projection)
                proj_dict = {f: 1 for f in proj_fields}
                proj_dict["_id"] = 1
                cursor = cursor.project(proj_dict)

            # For manual filtering (no target_fields), don't limit the cursor at all
            if target_fields and limit > 0:
                cursor = cursor.limit(limit)

            results = []
            
            # Create progress bar
            if target_fields:
                total_count = coll.count_documents(query)
                if limit > 0:
                    total_count = min(total_count, limit)
                pbar = tqdm(total=total_count, desc="Searching documents", disable=not show_progress)
            else:
                pbar = tqdm(desc="Scanning documents", disable=not show_progress)
            
            for doc in cursor:
                if target_fields:
                    results.append(doc)
                    pbar.update(1)
                else:
                    # Manual check across all string fields
                    matched = any(
                        self._is_string_field(value) and regex_pattern.search(value)
                        for value in doc.values()
                    )
                    if matched:
                        results.append(doc)
                        if limit > 0 and len(results) >= limit:
                            pbar.update(1)
                            break
                    pbar.update(1)
            
            pbar.close()

            print(f"\nFound {len(results)} matching document(s) in {database}.{collection}:\n")
            for i, doc in enumerate(results, 1):
                print(f"--- Result {i} ---")
                for k, v in doc.items():
                    if isinstance(v, str) and len(v) > 300:
                        print(f"{k}: {v[:297]}...")
                    else:
                        print(f"{k}: {v}")
                print()

        except OperationFailure as e:
            print(f"Query failed (auth/permission issue?): {e}")
        except Exception as e:
            print(f"Error during search: {e}")

    def close(self):
        self.client.close()
        print("Connection closed.")


def main():
    fire.Fire(MongoRegexSearch)


if __name__ == "__main__":
    main()