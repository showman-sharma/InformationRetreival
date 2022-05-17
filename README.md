# InformationRetreival
## Description
For any given query, an Information Retrieval (IR) system is used to obtain and rank relevant word documents from the data collection of interest. The most basic IR system uses Term Frequency Inverse Document Frequency (TF-IDF) to represent documents and queries as vectors, and then uses measures like cosine similarity to assess the relevance of a query to all the documents in the dataset.

This TF-IDF based Vector Space Model (VSM) performs admirably, but it has a few flaws, such as the assumption that the words are semantically unrelated, necessitating the development of alternate IR approaches. The NLP community has spent a lot of time looking into ways to capture semantic relatedness in words and documents. Data-driven learning of vector representations from a big corpus is one of the most common ways.

While bottom-up approaches have proven to be beneficial in recent years, the results of such models can be enhanced by including some type of top-down information accumulated over time by people. These top-down resources can be used to direct bottom-up excursions to find relationships between words and documents. Several new works in this field have lately surfaced. Latent Semantic Analysis is a well-known bottom-up approach for capturing semantic relatedness in words (LSA). Another method for capturing semantic relatedness in words/documents is explicit semantic analysis (ESA). ESA improves word representations by using Wikipedia's top-down knowledge.

## Usage
In main.py, kindly change line 5 accordingly before proceeding:
1. baseline: 
from informationRetrieval_baseline import InformationRetrieval
2. Limited words(LW): 
from informationRetrieval_LW import InformationRetrieval
3. LW+LSA: 
from informationRetrieval_LSA import InformationRetrieval
4. LW+ESA: 
from informationRetrieval_ESA2 import InformationRetrieval
5. LW+NESA: 
from informationRetrieval_NESA2 import InformationRetrieval
6. LW+bigrams: 
from informationRetrieval_bigram2 import InformationRetrieval
7. LW+ESA+LSA: 
from informationRetrieval_ESA import InformationRetrieval
8. LW+NESA+LSA: 
from informationRetrieval_NESA import InformationRetrieval
9. LW+LSA+bigrams: 
from informationRetrieval_bigram import InformationRetrieval


Run main.py as before with the appropriate arguments.
Usage: 
> python main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom

> Enter query below

> Papers on Aerodynamics

This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.
