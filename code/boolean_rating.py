import argparse
import pickle
import pandas as pd

# Function to load opinion lexicon files
def load_opinion_lexicon():
    """
    Load positive and negative opinion words from the lexicon files.
    """
    positive_file = 'positive-words.txt'
    negative_file = 'negative-words.txt'
    
    with open(positive_file, 'r') as f:
        positive_words = set(line.strip().lower() for line in f if line.strip())

    with open(negative_file, 'r') as f:
        negative_words = set(line.strip().lower() for line in f if line.strip())
    
    return positive_words, negative_words

# Function to load the posting list from posting_list.pkl
def load_posting_list():
    with open('output/posting_list.pkl', 'rb') as f:
        posting_list = pickle.load(f)
    return posting_list

# Boolean and Rating Search with Lexicon
def boolean_and_rating_search(posting_list, aspect1, aspect2, opinion1, method, positive_words, negative_words, opinion2=None):
    """
    Perform Boolean search augmented with rating and opinion lexicon:
    - Boolean operations: OR, AND, or (OR) AND based on method.
    - Filter reviews based on rating and opinion orientation using lexicon.
    """

    # Step 1: Perform Boolean Filtering
    if method == "method1":
        # OR operation: aspect1 OR aspect2 OR opinion1 OR (opinion2 if provided)
        intermediate_results = (set(posting_list.get(aspect1, [])) |
                                set(posting_list.get(aspect2, [])) |
                                set(posting_list.get(opinion1, [])))
        if opinion2:
            intermediate_results |= set(posting_list.get(opinion2, []))
    elif method == "method2":
        # AND operation: aspect1 AND aspect2 AND opinion1 AND (opinion2 if provided)
        intermediate_results = (set(posting_list.get(aspect1, [])) &
                                set(posting_list.get(aspect2, [])) &
                                set(posting_list.get(opinion1, [])))
        if opinion2:
            intermediate_results &= set(posting_list.get(opinion2, []))
    elif method == "method3":
        # (OR) AND operation: (aspect1 OR aspect2) AND (opinion1 OR opinion2 if provided)
        intermediate_results = ((set(posting_list.get(aspect1, [])) |
                                 set(posting_list.get(aspect2, []))) &
                                set(posting_list.get(opinion1, [])))
        if opinion2:
            intermediate_results |= ((set(posting_list.get(aspect1, [])) |
                                      set(posting_list.get(aspect2, []))) &
                                     set(posting_list.get(opinion2, [])))
    else:
        raise ValueError("Invalid method. Choose 'method1', 'method2', or 'method3'.")

    # Step 2: Apply Rating Filter Using Lexicon
    filtered_results = []
    for review_id, rating, _ in intermediate_results:
        try:
            rating = int(rating)  # Convert rating to an integer for comparison
        except ValueError:
            continue  # Skip invalid ratings

        # Determine opinion orientation using lexicon
        opinion1_match = opinion1.lower() in positive_words or opinion1.lower() in negative_words
        opinion2_match = opinion2 and (opinion2.lower() in positive_words or opinion2.lower() in negative_words)

        if opinion1_match:
            if opinion1.lower() in positive_words and rating > 3:
                filtered_results.append((review_id, rating))
            elif opinion1.lower() in negative_words and rating <= 3:
                filtered_results.append((review_id, rating))
        
        if opinion2 and opinion2_match:
            if opinion2.lower() in positive_words and rating > 3:
                filtered_results.append((review_id, rating))
            elif opinion2.lower() in negative_words and rating <= 3:
                filtered_results.append((review_id, rating))

    # Deduplicate results
    filtered_results = list(set(filtered_results))  # Remove duplicates

    return filtered_results

# Main function to parse arguments, run search, and save results
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Boolean and Rating Search with Lexicon Implementation.")
    parser.add_argument("--aspect1", type=str, required=True, help="First aspect word")
    parser.add_argument("--aspect2", type=str, required=True, help="Second aspect word")
    parser.add_argument("--opinion1", type=str, required=True, help="First opinion word")
    parser.add_argument("--opinion2", type=str, default=None, help="Optional second opinion word")
    parser.add_argument("--method", type=str, required=True, help="Search method (method1, method2, method3)")

    args = parser.parse_args()

    # Load the posting list (inverted index)
    posting_list = load_posting_list()

    # Load the opinion lexicon files
    positive_words, negative_words = load_opinion_lexicon()

    # Execute the Boolean and Rating Search with Lexicon
    results = boolean_and_rating_search(posting_list, args.aspect1, args.aspect2, args.opinion1, args.method, positive_words, negative_words, args.opinion2)

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["review_id", "rating"])

    # Display the DataFrame with review_id column only
    df = df.drop(columns=["rating"])  # Exclude rating from the final output
    print("\nDataFrame to be saved (showing only review_id):")
    print(df)

    # Save DataFrame to .pkl file with the required naming convention
    output_file = f"./output/{args.aspect1}_{args.aspect2}_{args.opinion1}_{args.opinion2 or 'NA'}_{args.method}_4_2_with_rating.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"\nResults saved to {output_file}")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
