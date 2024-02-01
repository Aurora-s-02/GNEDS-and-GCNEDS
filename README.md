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
