# Cluster Titles to Vec Converter

This script converts a `cluster_titles.json` file into `.vec` format by fetching article data from MongoDB.

## Overview

The script:
1. Reads the input JSON file containing article IDs organized by category
2. Connects to MongoDB using credentials from environment variables
3. Fetches article details (title, published date, source, bias, entities, text) for each article ID
4. Formats and writes the data in `.vec` format

## Prerequisites

- Python 3.x
- pymongo library: `pip install pymongo --break-system-packages`
- MongoDB credentials set in environment variables

## Environment Variables

You must set these environment variables before running:

```bash
export MONGO_USER="your_username"
export MONGO_PASS="your_password"
```

## Usage

```bash
python convert_to_vec.py <input_json> <output_directory>
```

### Example

```bash
python convert_to_vec.py cluster_titles.json ./vec_output
```

This will create a directory called `vec_output` containing one `.vec` file for each category.

## Input Format

The input JSON should have this structure:

```json
{
  "categories": [
    {
      "category_id": 1,
      "category_name": "Category Name",
      "article_count": 10,
      "articles": [
        {
          "article_id": "697cc33fb28564af3e4b89a8",
          "title": "Article title"
        }
      ]
    }
  ]
}
```

## Output Format

The script creates **one `.vec` file per category** in the output directory. Each file is named using the pattern:

```
<category_id>_<sanitized_category_name>.vec
```

For example:
- `001_Alcaraz_Open.vec`
- `002_Congo_200.vec`
- `003_Sentenced_Hasina.vec`

Each `.vec` file contains:

```
================================================================================
CATEGORY 1: Alcaraz / Open
Articles: 13
================================================================================

---
ID: 697cc33fb28564af3e4b89a8
Title: Article title
Published: 2026-02-02T13:49:30
Source: source-name
Bias: {"dir":{"L":0.05,"C":0.9,"R":0.05},"deg":{"L":0.8,"M":0.15,"H":0.05},"reason":"..."}
<entities>
DATE: Monday, last year
GPE: Morocco, Rome
PERSON: John Doe
</entities>
Text: The full article text goes here...

---
ID: 698120c4977a8442e6b01a6a
Title: Another article in same category
...
```

**File Naming:**
- Category IDs are zero-padded to 3 digits (001, 002, etc.) for proper sorting
- Special characters in category names (/, \, :, *, etc.) are replaced with underscores
- Spaces are replaced with underscores
- Multiple consecutive underscores are collapsed to single underscores

## Database Schema

The script expects MongoDB documents with this structure:

```json
{
  "_id": ObjectId("..."),
  "title": "Article title",
  "published": {"$date": "2026-02-02T13:49:30.000Z"},
  "source": "source-name",
  "bias": "{\"dir\":{\"L\":0.05,\"C\":0.9,\"R\":0.05},...}",
  "ner": {
    "entities": [
      {"text": "Rome", "label": "GPE"},
      {"text": "Monday", "label": "DATE"}
    ]
  },
  "article": "The full article text..."
}
```

Key field mappings:
- `published`: MongoDB date object (converted to ISO format)
- `bias`: JSON string (parsed and reformatted)
- `ner.entities`: Array of entity objects (grouped by label)
- `article`: The article text content

## Database Connection

The script uses the same MongoDB connection details as `mongo_rw.py`:
- Database: `rssnews`
- Collection: `articles`
- Connection: `mongodb://{user}:{pass}@localhost:27017`

## Features

- **Separate Files Per Category**: Each category gets its own `.vec` file for easy organization
- **Smart File Naming**: Files are named with zero-padded category IDs and sanitized category names
- **Category Headers**: Each file includes a header with category information
- Per-category summary shows processed and skipped counts
- Creates output directory automatically if it doesn't exist
- Skips articles that cannot be found in the database
- Handles missing fields gracefully
- Formats entities alphabetically by type
- Preserves bias data as JSON

## Error Handling

- Missing documents generate warnings but don't stop processing
- Connection errors will halt execution with an error message
- Progress summary shows processed and skipped article counts

## Notes

- The script expects article IDs in ObjectId format (24-character hexadecimal strings)
- All fields are retrieved directly from MongoDB except the article_id which comes from the input JSON
- Entities are formatted with each type on a separate line within the `<entities>` tags
