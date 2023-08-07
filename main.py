import base64

from huggingface_hub import hf_hub_download
from flair.models import TextClassifier
import flair
from flair.data import Sentence
import pandas as pd
import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_elements import elements, mui, html, nivo

st.set_page_config(layout="wide", initial_sidebar_state="expanded")



def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)






# set title of the app
st.title('Text Classification based on Sustainable Development Goals (SDGs): Classification / Explainability App')



#Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pipeline_showcase.py", "Showcase Pipeline", "🏗"),
        Page("plain_text.py", "Plain Text Input", "🔣"),
        Page("explainability_sdg_model.py", "SDG Explainability", "🧠"),
    ]
)

# Optional -- adds the title and icon to the current page

st.divider()
st.header("Project Description")
st.subheader("Background")
st.write("The Sustainable Development Goals (SDGs) are a collection of 17 global goals set by the United Nations General Assembly in 2015 for the year 2030. The SDGs are part of Resolution 70/1 of the United Nations General Assembly, the 2030 Agenda. The SDGs build on the principles agreed upon in Resolution A/RES/66/288, entitled \"The Future We Want\". This resolution was a broad intergovernmental agreement that acted as the precursor for the SDGs. The goals are broad-based and interdependent. The 17 sustainable development goals each have a list of targets that are measured with indicators. The total number of targets is 169. Below is an image of the 17 SDGs:")
st.image("media/Clariant Image SDG Poster 2021.jpg", width=700)
st.subheader("Project Goal")
st.write("The goal of this project is to classify text data into the 17 SDGs. The project will use 4 different datasets consisting of 65938 labeled sentences. The dataset is available on Huggingface. The project will use Natural Language Processing (NLP) techniques to classify the text data into the 17 SDGs. The project will use a transformer architecture combined with a classification layer to classify the textual data. Furthermore, the project will utilize a stratified K-fold approach along with the F1 score to evaluate the performance of the classification model and guarantee consistency throughout the unlabeled dataset.")
st.subheader("Project Motivation")
st.write("The motivation for this project is to use NLP techniques to classify text data into the 17 SDGs. Moreover, the project will develop a evaluation pipeline that will be applied on 50 scraped published sustainability reports from various goods/services companies.")
st.divider()
st.header("Datasets")
st.write("The project will use 4 different datasets consisting of 65938 labeled sentences. The dataset is available on Huggingface. The datasets are as follows:")
dataset_list = ["original_data.csv: 2567 labeled sentences (supplied by the Quality & Usability supervision team)",
                "politics.csv: 22977 labeled sentences (supplied by the Quality & Usability supervision team)",
                "targets.csv: 332 labeled sentences (supplied by the Quality & Usability supervision team)",
                "osdg_data.csv: 40062 labeled sentences (The OSDG Community Dataset (OSDG-CD))"]
for i in dataset_list:
    st.markdown("- " + i)
st.divider()
st.subheader("Dataset Distributions")
st.image(["media/single_dist.png", "media/merged_dist.png"], width=500)
st.divider()
st.header("Model")
st.write("The main adopted architecture for this project is the transformer model in combination with a classification layer. Multiple models were tested out on different combinations of the dataset, more on that can be found in the report later. The best performing model was the BERT base model (cased) trained on a stratified 5-fold merged variant of the dataset. The Model was trained using following hyperparameters:")
model_hyperparameters = ["Batch Size: 16",
                         "Learning Rate: 3e-5",
                         "Epochs: 4",
                         "Optimizer: AdamW",
                         "Scheduler: LinearSchedulerWithWarmup",
                         "Loss Function: CrossEntropyLoss",
                         "Evaluation Metric: Accuracy, F1 Score",
                         "Random Seed: 42",
                         "Stratified K-Fold: 5"
                         ]
for i in model_hyperparameters:
    st.markdown("- " + i)
st.divider()
st.header("Pipeline Flowchart Graph:")
displayPDF("media/pipeline_flowchart.pdf")




