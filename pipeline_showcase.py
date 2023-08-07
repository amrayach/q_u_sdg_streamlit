import base64

from flair.models import TextClassifier
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from st_pages import Page, show_pages, add_page_title
from st_aggrid.shared import ExcelExportMode

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def displayPDF(file, height=1000, width=600):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


add_page_title()
st.divider()

sing_scores = pd.read_excel('media/Company_Info.xlsx', sheet_name='Single Scores')
indev_scores = pd.read_excel('media/Company_Info.xlsx', sheet_name='Individual Scores')
company_desc = pd.read_excel('media/Company_Info.xlsx', sheet_name='Company ListDescription')

st.header('Company List Description')
st.write('The following table contains the meta data of the 50 companies that were scraped for this project. Moreover, '
         'a brief description of the company is provided. The most important information is the CO2 emissions per '
         'Revenue which is used in the final score calculation.')
AgGrid(company_desc, height=400)
st.write('The following chart depicts the CO2 emissions per Revenue for the 50 companies.')
displayPDF('media/CO2_Emission_per_Million_Euros.pdf', height=600, width=1000)
st.divider()
st.header('Individual Scores')
st.write('The following table depicts the single company SDG scores per company. '
         'The scores are calculated by the sum of the individual SDG scores divided by the number of SDGs.')
AgGrid(indev_scores, height=400)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.write('The following chart displays the company scores per SDG.')
option = st.selectbox(
    "Select a SDG to filer by:",
    ("SDG1", "SDG2", "SDG3",
     "SDG4", "SDG5", "SDG6",
     "SDG7", "SDG8", "SDG9",
     "SDG10", "SDG11", "SDG12",
     "SDG13", "SDG14", "SDG15",
     "SDG16", "SDG17"),
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

displayPDF("media/SDG_per_Company/" + option + ".pdf", height=1000, width=600)
st.divider()

company_list = tuple(['IAV.pdf',
                'Stadler.pdf',
                'AroundTown.pdf',
                'DeutscheKonsum.pdf',
                'Solaris.pdf',
                'Bosch.pdf',
                'Zalando.pdf',
                'Footprint.pdf',
                'Oatly.pdf',
                'Siemens.pdf',
                'Ecosia.pdf',
                'Fielmann.pdf',
                'Bombardier.pdf',
                'Veganz.pdf',
                'Volkswagen.pdf',
                'DeliveryHero.pdf',
                'Bertrandt.pdf',
                'BerlinHypAG.pdf',
                'ASML.pdf',
                'Babbel.pdf',
                'Auto1.pdf',
                'BASF.pdf',
                'DPDHL.pdf',
                'MTU.pdf',
                'Home24.pdf',
                'Edeka.pdf',
                'MesseBerlin.pdf',
                'SAP.pdf',
                'Vattenfall.pdf',
                'EON.pdf',
                'Adler.pdf',
                'Bayer.pdf',
                'ArcelorMittal.pdf',
                'VisitBerlin.pdf',
                'HelloFresh.pdf',
                'Allianz.pdf',
                'Goodyear.pdf',
                'BellevueInvestments.pdf',
                'CocaCola.pdf',
                'Helios.pdf',
                'MercedesBenz.pdf',
                'AxelSpringer.pdf',
                'Schindler.pdf',
                'Enterprise.pdf',
                'Pfizer.pdf',
                'FU.pdf',
                '12Tree.pdf',
                'Rewe.pdf',
                'MisterSpex.pdf',
                'DeutscheBahn.pdf'])

st.write('The following chart displays the SDG scores per company.')
option2 = st.selectbox(
    "Select a company to filter by:",
    company_list,
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

displayPDF("media/Company_per_SDG/" + option2, height=450, width=700)

st.divider()

st.write('This is the final scoring table. The final score is calculated by '
         'including the SDG, keywords and meta data as described in the report. ')
st.header('Single Scores')
AgGrid(sing_scores, height=400)
st.write('The following chart depicts the overall (sum of all SDGs) per company scores.')
displayPDF('media/sdg_per_sentence.pdf', height=500, width=1300)
st.divider()
st.write('The following chart depicts the average score per company after including all the scoring factors.')
displayPDF('media/Average_Score.pdf', height=500, width=1300)
st.divider()


# st.tabs(["Company 1", "Company 2", "Company 3"])
# tab1, tab2, tab3 = st.tabs(["Company 1", "Company 2", "Company 3"])
# with tab1:
# print()


