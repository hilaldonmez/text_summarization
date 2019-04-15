# Extractive Text Summarization for Single Document with LexRank

Single document extractive text summarization with LexRank is implemented. Dataset consists of 1000 plain text files. Each of them includes a news story and gold standard summary separated with a blank line from the news story. Implementation steps are as following:
1. Calculate IDF score
2. Build the graph
3. Evaluation with Rouge with the [library](https://github.com/pltrdy/rouge) 



How to run the program?

python3 lexRank.py dataset_folder_path file_name

Output is the LexRank score for each sentence in that file. 
