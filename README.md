
# NLU-PIPELINE

# OBJECTIVE
The aim of the project is to develop an automated **NATURAL LANGUAGE UNDERSTANDING PIPELINE** the technology behind conversational AI which inlcudes tasks like **intent classification**,**slot extraction**,aiming to provide a semantic way for user utterances.Intent classification focuses on the intent (context) of the query, while slot filling extracts semantic concepts in the user query.There are a total of 5 modules used in this Project:

1. Query correctness
2. Query Type
3. Intent classification
4. Slot extraction
5. Sentiment classification


# DATASET
All the tasks have been performed on a labelled **ATIS Dataset** for flights.

# IMPLEMENTATION

File init_pipeline.py contains all the modules required for running the project









## Run Locally

Clone the project

```bash
  git clone https://github.com/eashan27/NLU-PIPELINE.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Go to the project directory

```bash
  cd omni-channel/code/init_pipeline.py
```


## INPUT

```bash
USER UTTERANCE- I want to fly from baltimore to dallas round trip
```

## OUTPUT
Module 1:Query Correctness
```bash
correctness': {'value': 'correct', 'confidence': '0.9998168528035659', 'entity': 'query', 'extractor': 'query_correctness_extractor'}
```

Module 2: Query Type
 ```bash
 type': {'value': 'DESCRIPTION', 'confidence': '0.9966506122903696', 'entity': 'query', 'extractor': 'query_type_extractor'}
 ```
Module 3: Intent classification
```bash
intent': {'value': ['atis_flight'], 'confidence': '[0.99599236]', 'intent': 'multi_intent', 'extractor': 'multi_intent_extractor
```
Module 4:Slots
```bash
entities': {'slot': 'B-fromloc.city_name', 'value': 'baltimore', 'extractor': 'slot_extractor'}, {'slot': 'B-toloc.city_name', 'value': 'dallas', 'extractor': 'slot_extractor'{'slot': 'B-round_trip', 'value': 'round', 'extractor': 'slot_extractor'}, {'slot': 'I-round_trip', 'value': 'trip', 'extractor': 'slot_extractor'}}
```
Module 5:Sentiment Classification
```bash
sentiment': {'value': 'POSITIVE', 'confidence': 0.9926454424858093, 'entity': 'sentiment', 'extractor': 'sentiment_extractor
```

![Screenshot from 2021-12-31 03-24-30](https://user-images.githubusercontent.com/56325514/147815012-0381f304-1048-4190-96c1-10bf21e540c2.png)
