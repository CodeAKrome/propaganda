How It Works Now:

TF-IDF Vectorization: Converts article titles and content into numerical vectors that capture the importance of words
User Profile: Creates a weighted average profile from articles the user rated highly
Cosine Similarity: Measures how similar new articles are to the user's profile
Content-Based Filtering: Recommends articles with similar content to what the user liked

Key Features:

Title Weighting: Titles are repeated 3x in the text to give them more importance
User Profile Building: Combines all rated articles, weighted by rating score
Similarity Scoring: Returns articles most similar to user's preferences
Article Similarity: Can also find articles similar to a specific article

Main Methods:

add_rating() - Store user ratings
recommend_articles() - Get recommendations based on content similarity
recommend_with_details() - Get recommendations with full article info
get_similar_articles() - Find articles similar to a specific article

Example:
If a user rates political articles highly, the system will recommend other political articles with similar topics, keywords, and writing style based on the actual article content!
