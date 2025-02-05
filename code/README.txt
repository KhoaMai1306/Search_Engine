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