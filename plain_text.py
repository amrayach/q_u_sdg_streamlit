import streamlit as st
from flair.models import TextClassifier
from flair.data import Sentence
import spacy
from huggingface_hub import hf_hub_download
from st_pages import Page, show_pages, add_page_title
import pandas as pd
import plotly.graph_objects as go
import pickle


#sdg_keyword_dict = pickle.load(open('sdg_keywords_dict.pickle', 'rb'))
#print()


st.set_page_config(layout="wide", initial_sidebar_state="expanded")


add_page_title()
st.divider()

@st.cache_resource
def init_nlp_model():
    return spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "morphologizer",
                                            "tagger", "attribute_ruler"])

@st.cache_resource
def init_sdg_classifier_model():
    model_path = hf_hub_download(repo_id="amay01/quality_and_usability_sgd_17_classifier_flair",
                                 filename="final-model.pt",
                                 force_download=True)
    return TextClassifier.load(model_path)

#@st.cache_resource
#def init_sdg_keyword_dict():


nlp = init_nlp_model()
model = init_sdg_classifier_model()

st.write("This is a simple plain text demo tool for the SDG classifier. It is meant to be used for short texts, such as sentences or paragraphs. Once the text is entered, "
         "a Spacy sentence segmentation model splits the input into single sentences which the classifier predicts the SDG labels for. The results are then visualized as a pie chart and a table with the single sentence and respective label. "
         " Moreover, similar to the classification, a keyword detection is performed for each sentence. The keywords are then visualized as a pie chart as well.")
st.divider()

txt = st.text_area('🖹 Text to classify for SDGs:', '''Encouraging suppliers to adopt the above standards. Disclosing industrial accidents and cases of occupational disease. Providing innovative solutions to improve the access and quality of health services in remote areas.''')

curr_doc = nlp(txt)
text_sents = [sent.text for sent in curr_doc.sents]


# create a dictionary and fill it with sdg 0 to 16 as keys and 0 as values
sdg_dict = {'no-sdg': 0}
for i in range(17):
    sdg_dict["sdg-"+str(i)] = 0


sent_prediction = []

for sent in text_sents:
    print("Processing sentence: ", sent)
    curr_sent = Sentence(sent)
    model.predict(curr_sent)

    if curr_sent.labels:
        if curr_sent.score <= 0.55:
            sent_prediction.append([curr_sent.text, "NO-SDG"])
            sdg_dict['no-sdg'] += 1
        else:
            sent_prediction.append([curr_sent.text, curr_sent.tag.replace("__label__", "SDG ")])
            sdg_dict[curr_sent.tag.replace("__label__", "sdg-")] += 1
    else:
        sent_prediction.append([curr_sent.text, "NO-SDG"])
        sdg_dict['no-sdg'] += 1


sdg_labels = []
sdg_values = []
for key, value in sdg_dict.items():
    if value > 0:
        sdg_labels.append(key)
        sdg_values.append(value)


fig = go.Figure(data=[go.Pie(labels=sdg_labels, values=sdg_values, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])

st.plotly_chart(fig, use_container_width=True)

df = pd.DataFrame(sent_prediction, columns=['sentence', 'sdg_label'])

st.text("📊 Results:")
st.dataframe(df)


# https://github.com/sadickam/sdg-classification-bert/blob/main/main.py
# print()

