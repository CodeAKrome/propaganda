#!/usr/bin/env python3
"""
MongoDB field reader/writer tool using Fire library.

Usage:
  python mongo_tool.py read --id=<doc_id> --field=<field_name>
  python mongo_tool.py write --id=<doc_id> --field=<field_name> --data=<value> [--force]
"""

import os
import sys
import json
import fire
from pymongo import MongoClient
from bson.objectid import ObjectId


class MongoFieldTool:
    """Tool for reading and writing individual fields in MongoDB documents."""

    def __init__(self):
        """Initialize MongoDB connection."""
        mongo_user = os.getenv("MONGO_USER")
        mongo_pass = os.getenv("MONGO_PASS")

        if not mongo_user or not mongo_pass:
            raise ValueError(
                "MONGO_USER and MONGO_PASS environment variables must be set"
            )

        uri = f"mongodb://{mongo_user}:{mongo_pass}@localhost:27017"
        self.client = MongoClient(uri)
        self.db = self.client["rssnews"]
        self.collection = self.db["articles"]

    def read(self, id: str, field: str):
        """
        Read a specific field from a document.

        Args:
            id: MongoDB document ID
            field: Field name to read

        Returns:
            The value of the specified field
        """
        try:
            doc = self.collection.find_one({"_id": ObjectId(id)})

            if doc is None:
                print(f"Error: Document with ID '{id}' not found")
                return None

            if field not in doc:
                print(f"Error: Field '{field}' not found in document")
                return None

            value = doc[field]

            # If value is dict or list, output as JSON only
            if isinstance(value, (dict, list)):
                print(json.dumps(value))
            else:
                print(value)

            return None

        except Exception as e:
            print(f"Error reading field: {e}")
            return None

    def write(self, id: str, field: str, data: str, force: bool = False):
        """
        Write data to a specific field in a document.

        Args:
            id: MongoDB document ID
            field: Field name to write
            data: Data to write to the field (use '-' to read from stdin)
            force: If True, overwrite existing data; if False, skip if field has data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read from stdin if data is '-'
            if data == "-":
                data = sys.stdin.read().strip()
                if not data:
                    print("Error: No data received from stdin")
                    return False

            # If not forcing, check if field already has data using projection
            if not force:
                doc = self.collection.find_one(
                    {"_id": ObjectId(id)},
                    {field: 1}  # Only fetch the specific field
                )
                
                if doc is None:
                    print(f"Error: Document with ID '{id}' not found")
                    return False

                # Check if field exists and has data (not None and not empty string)
                if field in doc and doc[field] not in (None, ""):
                    print(f"Skipped: Field '{field}' already has data. Use --force to overwrite.")
                    return False

            # For bias field, parse JSON string and store as object
            if field == "bias":
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    # If not valid JSON, store as-is (legacy behavior)
                    pass

            # Perform the update
            result = self.collection.update_one(
                {"_id": ObjectId(id)}, {"$set": {field: data}}
            )

            if result.matched_count == 0:
                print(f"Error: Document with ID '{id}' not found")
                return False

            if result.modified_count > 0:
                print(f"Successfully updated field '{field}'")
            else:
                print(f"Field '{field}' already had this value (no change made)")

            return True

        except Exception as e:
            print(f"Error writing field: {e}")
            return False


def main():
    """Main entry point for Fire CLI."""
    fire.Fire(MongoFieldTool())


if __name__ == "__main__":
    main()
