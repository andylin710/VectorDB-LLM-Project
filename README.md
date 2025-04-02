# VectorDB-LLM-Project

## Introduction
This repository contains code for testing out the possible configurations in building a Retrieval-Augmented system that would answer queries based on input documents. 

### Procedure
- Sample documents (in this case, lecture slides) are read and preprocessed
- The preprocessed tokens are separated into chunks with variable amounts of overlapping
- An embedding model transforms each chunk into an embedding
- Each embedding is added to a key-value database, with the keys being the chunk origin and the values being the chunk emebddings
- A sample query is transformed to an embedding as well, as the top k chunk embeddings most similar to the query embedding is identified
- A query to an LLM model is sent with the context of the most similar embeddings
- The LLM model returns a user-friendly answer. If the corpus does not contain relevant information, the mdoel will say it doesn't have the necessary context

### Analysis
We outlined a few variables that could be modified for each configuration:
- Number of tokens within each chunk (200, 500, 1000)
- Number of tokens overlapping between chunks (0, 50, 100)
- Embedding model used (mpnet-base, minilm, nomic-embedded)
- LLM model used (llama3.2:latest, gemma3:latest)
- Database (redis, chromadb, qdrant)

We used these metrics to analyze the performance of each possible configuration: 
- Time it takes to full add all chunk embeddings into database
- Peak memory usage during database addition process
- Average time it takes to complete a query
- Overall quality of response

The same 15 queries (https://docs.google.com/document/d/15WQqcMf_-oeXZvPSYXQDC-lqHayPv4epNLhgtpKRMuE/edit?tab=t.0) were used for each configuration. All possible configurations were tested on the redis database. To compare the different databases, we used the llama3.2:latest LLM model and the all-mpnet-base-v2 embedding model. Not all possible configurations were tested across all variables in the interest of time.   

## Directory
### Notable Folders:
- All_Slides: Folder holding all of the documents to be ingested
- chroma_db: Helper files for Chroma database
- Results: Query and code outputs. Includes all of the query outputs for each specific configuration, as well as the combined output of the driver code. 

### Notable Files:
The following files contain the testing stack used by the driver file that implements the entire procedure for a specific database
- chromaDB_pipeline.py: Chroma database functions
- qdrant_pipeline.py: Qdrant database functions
- redis_pipeline.py: Redis database functions

The following files contain the test results done on a specific computer to minimize differences based on operating systems and computer specs:
- roland_chroma_test_results.csv
- roland_qdrant_test_results.csv
- roland_redis_test_results - Copy.csv

The following files are the driver / helper functions that act as our main testing functions. 
- driver.ipynb: Imports the necessary function(s) from one of the pipeline files, and runs a collection of configurations for the entire stack. Query results and metrics are recorded into the Results folder and the respective csv. 
- Visuals.ipynb: Creates visualizations to showcase the performance difference in the different configurations. 

