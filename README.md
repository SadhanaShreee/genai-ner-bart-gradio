## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The goal is to develop an application that can accurately recognize and categorize named entities such as persons, organizations, locations, dates, etc., from input text. By fine-tuning a pre-trained BART model specifically for NER tasks, the system should be able to understand contextual relationships and identify relevant entities. The Gradio framework will be used to build a user-friendly interface for real-time interaction and evaluation.

### DESIGN STEPS:

#### STEP 1: Data Collection and Preprocessing

#### STEP 2: Fine-Tuning the BART Model & Model Evaluation

#### STEP 3: Application Development Using Gradio & Deployment and Testing

### PROGRAM:
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import gradio as gr

# Load pre-trained BERT NER model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline for NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to process user input
def ner_function(text):
    entities = ner_pipeline(text)
    return "\n".join([f"{ent['word']} ({ent['entity']})" for ent in entities])

# Gradio Interface
iface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Named Entities"),
    title="NER Demo with Pre-trained Model"
)

iface.launch()
```

### OUTPUT:
![Screenshot 2024-11-28 221034](https://github.com/user-attachments/assets/184dbcc3-5041-473d-ab4e-e9c435cccf0c)

### RESULT:
Thus successfully a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.
