import argparse
import pickle
import pandas as pd

# Function to load the posting list from posting_list.pkl
def load_posting_list():
    with open('output/posting_list.pkl', 'rb') as f:
        posting_list = pickle.load(f)
    return posting_list

# Boolean search methods
def method1(posting_list, aspect1, aspect2, opinion1, opinion2=None):
    """
    Perform OR operation on terms: (aspect1 OR aspect2 OR opinion1 OR opinion2)
    If opinion2 is None, it performs OR on aspect1, aspect2, and opinion1 only.
    """
    if opinion2:
        result = (set(posting_list.get(aspect1, [])) |
                  set(posting_list.get(aspect2, [])) |
                  set(posting_list.get(opinion1, [])) |
                  set(posting_list.get(opinion2, [])))
    else:
        result = (set(posting_list.get(aspect1, [])) |
                  set(posting_list.get(aspect2, [])) |
                  set(posting_list.get(opinion1, [])))
    return list(result)

def method2(posting_list, aspect1, aspect2, opinion1, opinion2=None):
    """
    Perform AND operation on terms: (aspect1 AND aspect2 AND opinion1 AND opinion2)
    If opinion2 is None, it performs AND on aspect1, aspect2, and opinion1 only.
    """
    if opinion2:
        result = (set(posting_list.get(aspect1, [])) &
                  set(posting_list.get(aspect2, [])) &
                  set(posting_list.get(opinion1, [])) &
                  set(posting_list.get(opinion2, [])))
    else:
        result = (set(posting_list.get(aspect1, [])) &
                  set(posting_list.get(aspect2, [])) &
                  set(posting_list.get(opinion1, [])))
    return list(result)

def method3(posting_list, aspect1, aspect2, opinion1, opinion2=None):
    """
    Perform (aspect1 OR aspect2) AND (opinion1 OR opinion2)
    If opinion2 is None, it performs (aspect1 OR aspect2) AND opinion1 only.
    """
    if opinion2:
        result = ((set(posting_list.get(aspect1, [])) |
                   set(posting_list.get(aspect2, []))) &
                  (set(posting_list.get(opinion1, [])) |
                   set(posting_list.get(opinion2, []))))
    else:
        result = ((set(posting_list.get(aspect1, [])) |
                   set(posting_list.get(aspect2, []))) &
                  set(posting_list.get(opinion1, [])))
    return list(result)

# Main function to parse arguments, run search, and save results
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Boolean search implementation with optional second opinion.")
    parser.add_argument("--aspect1", type=str, required=True, help="First aspect word")
    parser.add_argument("--aspect2", type=str, required=True, help="Second aspect word")
    parser.add_argument("--first_opinion", type=str, required=True, help="First opinion word")
    parser.add_argument("--second_opinion", type=str, default=None, help="Optional second opinion word")
    parser.add_argument("--method", type=str, required=True, help="Search method (method1, method2, method3)")

    args = parser.parse_args()

    # Load the posting list (inverted index)
    posting_list = load_posting_list()

    # Execute the specified search method
    if args.method == "method1":
        results = method1(posting_list, args.aspect1, args.aspect2, args.first_opinion, args.second_opinion)
    elif args.method == "method2":
        results = method2(posting_list, args.aspect1, args.aspect2, args.first_opinion, args.second_opinion)
    elif args.method == "method3":
        results = method3(posting_list, args.aspect1, args.aspect2, args.first_opinion, args.second_opinion)
    else:
        raise ValueError("Invalid method. Choose 'method1', 'method2', or 'method3'.")

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["review_id", "rating", "cleaned_text"]).drop(columns=["rating","cleaned_text"])

    # Display the DataFrame with review_id and rating columns
    print("\nDataFrame to be saved (showing review_id):")
    print(df)

    # Save DataFrame to .pkl file with the required naming convention
    output_file = f"./output/{args.aspect1}_{args.aspect2}_{args.first_opinion}_{args.second_opinion or 'NA'}_{args.method}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)

    print(f"\nResults saved to {output_file}")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
