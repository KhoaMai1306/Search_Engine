import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_of_embeddings import get_word_embedding
import pickle

def load_posting_list():
    """Load the posting list (inverted index) from a prebuilt file."""
    with open('output/posting_list.pkl', 'rb') as f:
        posting_list = pickle.load(f)
    return posting_list

def embedding_search(query_aspects, query_opinions, method, embedding_method="glove", threshold=0.5):
    """
    Perform embedding-based search for reviews.

    Args:
        query_aspects (list): List of aspect keywords.
        query_opinions (list): List of opinion keywords.
        method (str): Search method (method1, method2, method3).
        embedding_method (str): Word embedding method to use ('glove', 'word2vec', etc.).
        threshold (float): Cosine similarity threshold for matching.

    Returns:
        DataFrame: Matching reviews.
    """
    # Load embeddings
    embedding_dict = get_word_embedding(embedding_method)

    # Generate embeddings for query terms
    aspect_embeddings = [
        embedding_dict[aspect] for aspect in query_aspects if aspect in embedding_dict
    ]
    opinion_embeddings = [
        embedding_dict[opinion] for opinion in query_opinions if opinion in embedding_dict
    ]

    # If any aspect or opinion embedding is missing, raise an error
    if not aspect_embeddings or not opinion_embeddings:
        raise ValueError("One or more query terms not found in the embedding dictionary!")

    # Compute the mean embeddings for aspects and opinions
    mean_aspect_embedding = np.mean(aspect_embeddings, axis=0).reshape(1, -1)
    mean_opinion_embedding = np.mean(opinion_embeddings, axis=0).reshape(1, -1)

    # Load posting list
    posting_list = load_posting_list()

    # Flatten all reviews in the posting list
    review_data = []
    for term in query_aspects + query_opinions:
        if term in posting_list:
            for review_id, _, review_text in posting_list[term]:  # Ignore review_rating
                review_data.append((review_id, review_text))

    # Remove duplicates
    unique_reviews = pd.DataFrame(
        review_data, columns=["review_id", "review_text"]
    ).drop_duplicates(subset=["review_id"])

    # Generate embeddings for each review text
    review_embeddings = []
    for text in unique_reviews["review_text"]:
        tokens = text.split()
        embeddings = [embedding_dict[word] for word in tokens if word in embedding_dict]
        if embeddings:
            review_embeddings.append(np.mean(embeddings, axis=0))
        else:
            review_embeddings.append(None)

    unique_reviews["embedding"] = review_embeddings

    # Drop rows with missing embeddings
    unique_reviews = unique_reviews.dropna(subset=["embedding"])

    # Calculate cosine similarity for each review
    similarities_aspect = cosine_similarity(
        np.vstack(unique_reviews["embedding"].to_numpy()), mean_aspect_embedding
    )
    similarities_opinion = cosine_similarity(
        np.vstack(unique_reviews["embedding"].to_numpy()), mean_opinion_embedding
    )

    # Combine similarities based on the chosen method
    if method == "method1":  # OR
        scores = np.maximum(similarities_aspect, similarities_opinion).flatten()
    elif method == "method2":  # AND
        scores = np.minimum(similarities_aspect, similarities_opinion).flatten()
    elif method == "method3":  # (OR) AND
        scores = (
            0.5 * np.maximum(similarities_aspect, similarities_opinion) +
            0.5 * np.minimum(similarities_aspect, similarities_opinion)
        ).flatten()
    else:
        raise ValueError("Invalid method. Choose method1, method2, or method3.")

    # Filter results based on threshold
    unique_reviews["score"] = scores
    filtered_reviews = unique_reviews[unique_reviews["score"] >= threshold]

    # Return sorted results
    return filtered_reviews.sort_values(by="score", ascending=False)

if __name__ == "__main__":
    # Prompt user for input
    query_aspects = input("Enter aspects (comma-separated): ").split(",")
    query_opinions = input("Enter opinions (comma-separated): ").split(",")
    method = input("Enter method (method1, method2, method3): ").strip()
    threshold = float(input("Enter similarity threshold (default 0.5): ").strip() or 0.5)

    # Run embedding search
    print("\nProcessing search query...")
    results = embedding_search(query_aspects, query_opinions, method, embedding_method="glove", threshold=threshold)

    # Reset index for proper ordering
    results = results.reset_index(drop=True)
    results.index += 1  # Start index from 1

    # Display all results with review_id and score
    print("\nDisplaying all matching reviews (if available):")
    print(results[["review_id", "score"]])

    # Save all results to an output file
    output_file = f"./output/embedding_{'_'.join(query_aspects)}_{'_'.join(query_opinions)}_{method}.pkl"
    results[["review_id", "score"]].to_pickle(output_file)
    print(f"\nFull results saved to {output_file}")
