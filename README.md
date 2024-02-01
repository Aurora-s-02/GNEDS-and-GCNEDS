# Event Type and Relationship Extraction Based on Dependent Syntactic Semantic Augmented Graph Networks

## Brief Introduction
In this study, two models, GNEDS and GCNEDS, are proposed to solve the trigger word and event type extraction task and the cross-sentence event relationship extraction task in documents, respectively. Dependent syntactic analysis is mainly utilized to enrich the intra-sentence semantic representation for the trigger word and event type extraction task, and then document nodes and sentence nodes are constructed to complete the cross-sentence event relation extraction task in documents.

## Project Structure
The structure of data and code is as follows:
```
|-- Datasets
|   |-- MAVEN_ERE
|   |   |-- map_docid_to_mentionids.json
|   |   |-- map_exid_to_docid.json
|   |   |-- mav_test.jsonl
|   |   |-- mav_train.jsonl
|   |   |-- mav_valid.jsonl
|   |-- MAVEN_ERE.zip
|   |-- OntoEvent-Doc
|   |   |-- OutoEvent_dependent
|   |   |   |-- data_on_doc_test_dependent.json
|   |   |   |-- data_on_doc_train_dependent.json
|   |   |   |-- data_on_doc_valid_dependent.json
|   |   |   |-- test_tokens_dependent.json
|   |   |   |-- train_tokens_dependent.json
|   |   |   `-- valid_tokens_dependent.json
|   |   |-- event_dict_label_data.json
|   |   |-- event_dict_on_doc_test.json
|   |   |-- event_dict_on_doc_train.json
|   |   `-- event_dict_on_doc_valid.json
|   |-- README.md
|   `-- __MACOSX
|       |-- MAVEN_ERE
|       `-- OntoEvent-Doc
|-- GCNEDS
|   |-- gcneds_distilbert.py
|   `-- run_gcneds.py
|-- GNEDS
|   |-- gneds_distilbert.py
|   `-- run_gneds.py
|-- data_utils.py
|-- requirements.txt
`-- run.sh
```
## Requirements 
- apex==0.9.10dev
- dgl==1.1.2+cu118
- ptvsd==4.3.2
- scikit_learn==1.3.1
- torch==2.1.0+cu118
- torchmetrics==0.9.3
- tqdm==4.66.1
- transformers==4.31.0

## Usage
### Project Preparation:
Download this project and unzip the dataset. You can directly download the archive, or run in your teminal.``` git clone https://github.com/Aurora-s-02/GNEDS-and-GCNEDS ```
### Data Preparation:
Unzip MAVEN_ERE and OntoEvent-Doc datasets stored at. ``` ./Datasets ```

