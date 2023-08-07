import streamlit as st
from flair.models import TextClassifier
from flair.data import Sentence
import spacy
from huggingface_hub import hf_hub_download
from st_pages import Page, show_pages, add_page_title
import pandas as pd
from flair.models import TextClassifier
import flair
from flair_model_wrapper import ModelWrapper
from interpret_flair import interpret_sentence, visualize_attributions
from captum.attr import LayerIntegratedGradients
import pickle
from captum.attr import visualization as viz
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


add_page_title()
st.divider()

@st.cache_resource
def init_sdg_classifier_model():
    model_path = hf_hub_download(repo_id="amay01/quality_and_usability_sgd_17_classifier_flair", filename="final-model.pt")
    return TextClassifier.load(model_path)


@st.cache_resource
def init_flair_xai_wrapper(_sdg_model):
    return ModelWrapper(_sdg_model)

@st.cache_resource
def init_layer_integrated_gradients_wrapper(_flair_xai_wrapper):
    return LayerIntegratedGradients(_flair_xai_wrapper, _flair_xai_wrapper.model.embeddings)



sdg_classifier = init_sdg_classifier_model()
#flair_model_wrapper = init_flair_xai_wrapper(sdg_classifier)
#lig = init_layer_integrated_gradients_wrapper(flair_model_wrapper)

st.write("This is a demo of the explainability of the SDG classifier. The Captum library is used to approximate the integral of the attributions. The approximations are calculated via the LayerIntegratedGradients method. "
         "The demo only takes a single sentence with its corresponding true label. Moreover, the user can specify the methods hyperparameters, but the default values should be sufficient for most use cases. "
         "Finally, once the results are computed a token attribution score is visualized as a heatmap and underneath the heatmap the token attributions are shown as a table.")
st.divider()

with st.form(key="my_form"):

    sentence = st.text_area('🖹 One sentence to explain SDG classification: ', '''As such there should be concerted efforts on part of all stakeholders and government officials for a paradigm shift to clean energy technologies by substituting the countries’ share of their energy mix from conventional energy of fossil fuel to clean energy sources.''')



    target_num = st.number_input('🔢 Enter true SDG label of the input sentence (0-16)',
                                 min_value=0,
                                 max_value=16,
                                 value=6,
                                 step=1
                                )

    n_steps = st.slider('🔢 The number of steps used by the approximation method (higher steps ~ higher computation time):',
                        min_value=50,
                        max_value=10000,
                        value=50,
                        step=100)

    estimation_method = st.selectbox('⇳ Method for approximating the integral:',
                                     options=['gausslegendre', 'riemann_right',
                                              'riemann_left', 'riemann_middle',
                                              'riemann_trapezoid'],
                                     index=0)

    submitted = st.form_submit_button(label="👉 Get SDG prediction / explanation !")

if submitted:
    flair_model_wrapper = ModelWrapper(sdg_classifier)
    lig = LayerIntegratedGradients(flair_model_wrapper, flair_model_wrapper.model.embeddings)



    visualization_list = []

    readable_tokens, word_attributions, delta = interpret_sentence(flair_model_wrapper,
                                                                    lig,
                                                                    sentence,
                                                                    '__label__' + str(target_num),
                                                                    visualization_list,
                                                                    n_steps=n_steps,
                                                                    estimation_method=estimation_method,
                                                                    internal_batch_size=3)


    attributions = visualize_attributions(visualization_list)


    st.markdown(attributions.data, unsafe_allow_html=True)

    st.write("  ")
    st.write("Table: Token Relevant Scores(-/+):")

    word_scores = word_attributions.detach().numpy()
    ordered_lig = [[readable_tokens[i], word_scores[i]] for i in np.argsort(word_scores)][::-1]
    df = pd.DataFrame(ordered_lig, columns=['Token', 'Importance-Score'])
    st.dataframe(df)






