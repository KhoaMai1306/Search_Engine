NLP-Based Search Engine for Filtering Customer Feedback
Project Overview
This project implements a Natural Language Processing (NLP) Search Engine that filters and retrieves user reviews based on specific aspect-opinion pairs using three different methods:

Boolean Search - Uses logical operations to filter relevant reviews.
Boolean Search with Rating Filtering - Enhances Boolean Search by considering customer ratings.
Embedding-based Search - Uses word embeddings to find semantically similar reviews.
The system builds an inverted index from a dataset of customer reviews (originally in an .xlsx file, but not uploaded). The search engine enables precise filtering of feedback, helping businesses extract valuable insights.

Real-World Applications
This project has practical applications in various industries, including e-commerce, customer service, and product development:

Amazon Review Filtering
Helps customers find specific feedback about a product (e.g., "battery life is poor", "screen quality is excellent").
Assists businesses in identifying common issues with their products by searching for frequently mentioned negative opinions.
Online Retail & E-commerce Analytics

Companies like Walmart, Best Buy, eBay can use this approach to filter user reviews and analyze customer sentiment on specific product features.
Helps retailers adjust their product descriptions based on commonly discussed features.
Tech Support & Customer Service Automation

Companies can filter customer complaints based on keywords and prioritize critical support tickets (e.g., "WiFi not working", "app crashes on startup").
Reduces manual effort in analyzing customer issues.

Hotel & Restaurant Review Analysis
Platforms like TripAdvisor, Yelp, and Google Reviews can extract insights such as "staff was friendly", "room was noisy", "food quality was bad", helping businesses improve their services.
Healthcare & Pharmaceutical Industry

Can be used to analyze patient reviews on medical services or prescription drugs, detecting common concerns (e.g., "side effects of medicine X", "doctor's attitude was unprofessional").
Social Media Monitoring & Brand Reputation

Brands can track public sentiment by filtering feedback from platforms like Twitter, Reddit, or forums, detecting issues such as "product delivery was delayed", "customer service was helpful".
Project Structure
The project consists of multiple Python scripts handling different aspects of the search engine:

1. Build Inverted Index (build_inverted_index.py)
Preprocesses the dataset, tokenizes reviews, removes stopwords, and lemmatizes words.
Creates an inverted index that maps keywords (aspects/opinions) to review IDs.
2. Boolean Search (boolean_search_help.py)
Implements three Boolean search methods:
Method 1 (OR Search): Returns reviews that mention any of the specified aspects or opinions.
Method 2 (AND Search): Returns reviews that contain all the specified aspects and opinions.
Method 3 (Hybrid Search): (aspect1 OR aspect2) AND (opinion1 OR opinion2).
3. Boolean Search with Rating Filter (boolean_rating.py)
Extends Boolean Search by filtering reviews based on ratings.
Uses an opinion lexicon to classify opinions as positive or negative.
Only returns reviews where sentiment aligns with the given rating threshold.
4. Embedding-Based Search (embedding_search.py)
Uses word embeddings (GloVe, Word2Vec, FastText, BERT) to retrieve reviews semantically similar to the query.
Uses cosine similarity to rank results based on how closely they match the input query.

Future Enhancements
Add topic modeling to categorize reviews into broader themes.
Improve sentiment analysis by fine-tuning a BERT model.
Build a web-based interface for an interactive search experience.
This README makes your project practical by tying it to real-world use cases and explaining how it can benefit businesses. 


Steps to Set Up the Environment
1. Activate Virtual Environment

2. Install Required Libraries
Run the following commands to install the necessary Python libraries:
pip install pandas requests nltk

3. Download NLTK Resources
After installing the nltk library, you need to download its required resources (e.g., stopwords, punkt, wordnet).

Option 1: Using Python Interactive Mode
Start the Python interpreter:
In the Python shell, run the following commands:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
Exit the Python shell by typing:
python
Copy code
exit()

Option 2: Using a Python Script
Create a file named download_nltk.py and add the following code:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
Save the file and execute it:
python download_nltk.py
4. Verify Setup
Run the script to ensure everything is working correctly:


python C:\Users\User\Desktop\COSC4397\code\build_inverted_index.py
Troubleshooting
If you encounter a ModuleNotFoundError, ensure you have activated the virtual environment and installed the required libraries inside it.
For PowerShell users, you may need to bypass execution restrictions temporarily by running:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then for boolean_search examples : 
For baseline:
        python C:\Users\User\Desktop\COSC4397\code\boolean_search_help.py --aspect1 audio --aspect2 quality --first_opinion poor --method method1   
    with 2 opinions:
        python C:\Users\User\Desktop\COSC4397\code\boolean_search_help.py --aspect1 mouse --aspect2 button --first_opinion click --second_opinion problem --method method2

        
For 4.2_boolean( which is boolean with rating filfer) :
python C:\Users\User\Desktop\COSC4397\code\4.2_boolean.py --aspect1 audio --aspect2 quality --opinion1 poor --method method1
python C:\Users\User\Desktop\COSC4397\code\4.2_boolean.py --aspect1 mouse --aspect2 button --opinion1 click --opinion2 problem --method method2


For embedding_search_help.py:
run the following command: 
pip install numpy pandas tqdm torch gensim transformers
pip install scikit-learn

then run the following command:
python C:\Users\User\Desktop\COSC4397\code\embedding_search.py 
Then it will run out the input from user format based one your queries.

note: the method 1 will be OR for all terms
method 2 will be AND for all terms
method 3 will be (aspect1 or aspect2) and (opinion1 or opinion2)
