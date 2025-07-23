import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv
import os


class HrLines:
    @staticmethod
    def hrV(height="200px"):
        width = "2px"
        st.markdown(
            f"""
            <div style='
                width: {width};
                height: {height};
                background-color: white;
                box-shadow: 0px 0px 10px white;
                margin-top: 2%;
                margin-bottom: 2%;
            '></div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def hrH(width="100%"):
        height = "2px"
        st.markdown(
            f"""
            <div style='
                width: {width};
                height: {height};
                background-color: white;
                box-shadow: 0px 0px 10px white;
                margin-top: 2%;
                margin-bottom: 2%;
            '></div>
            """,
            unsafe_allow_html=True
        )


load_dotenv()

try:
    df = pd.read_csv("datas/Dataset.csv")
except Exception:
    url = "https://drive.google.com/uc?id=1P7m86tazukZro1GKYa33SzJJA01LmrAu&export=download"
    df = pd.read_csv(url)
df = df.reset_index()

df["View"] = df["View"].fillna("Nowhere")
df["Furnishing_Status"] = df["Furnishing_Status"].fillna("Nothing")

df.drop(columns=["index", "No"], inplace=True)

df2 = df.sample(n=1000, random_state=42)

cities = df["Location"].unique()
uniqCities = pd.DataFrame(cities, columns=["Location"])
uniqCities.insert(0, 'Index', range(1, len(uniqCities) + 1))

hr = HrLines()

st.set_page_config(layout="wide")

st.markdown("<h1 style='color: purple; text-align: center;  font-size: 65px;'>House prices data engineering</h1>",
            unsafe_allow_html=True)

st.markdown(
    "<div style='color: orange; font-size: 20px;text-align: center;'>This is the capstone project of Zekeriya Barış ATAY, Software Engineering Department, MAKU, Burdur",
    unsafe_allow_html=True)

hr.hrH()

selected = option_menu(
    menu_title=None,
    options=["Database", "Analytics", "Data set information",
             "Prediction Panel"],
    icons=["database", "bar-chart", "info", "graph-up", "code"],
    menu_icon="cast",
    orientation="horizontal",
    styles={
        "icon": {"color": "purple", "font-size": "25px"},

        "container": {"padding": "15px", "gap": "10px"},

        "nav-link": {"font-size": "16px", "text-align": "center",
                     "justify-content": "center", "margin-right": "15px"},

        "nav-link-selected": {"background-color": "grey",
                              "box-shadow": "0 0 10px purple"}
    }
)
if selected == "Database":

    st.subheader("A small random subset of the database")
    st.write("---")
    st.dataframe(df2, height=570)
    st.title("Data Assistant")

    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])
    if "last_user_msg" not in st.session_state:
        st.session_state.last_user_msg = ""
    if "last_ai_msg" not in st.session_state:
        st.session_state.last_ai_msg = ""

    st.text_area(
        "Gemini flash 2.5:",
        value=st.session_state.last_ai_msg,
        height=200,
        disabled=True
    )

    user_input = st.text_area(
        "Write a message",
        value=st.session_state.last_user_msg,
        height=100,
        max_chars=1000,
    )

    if st.button("Submit"):
        if user_input.strip():
            st.session_state.chat.send_message(user_input)
            response = st.session_state.chat.last.text
            st.session_state.last_user_msg = ""
            st.session_state.last_ai_msg = response
            st.rerun()

elif selected == "Data set information":
    st.subheader("Information about this dataset")
    st.write("---")
    st.write(
        "This dataset is about house prices. It includes basic house information which is taken from [SamWash94]"
        "(https://www.kaggle.com/datasets/samwash94/dataset-for-house-price-analysis) on Kaggle.")
    count = len(df)
    formatted = f"{count:,}"
    st.write(
        f"The dataset contains {formatted} rows and {len(df.columns) - 1} columns. This consists of data collected for the year 2024.")
    st.write("---")
    col11, col22 = st.columns(2)
    with col11:

        st.write("The data includes some cities in USA")
        citiesdf = pd.DataFrame(df["Location"].unique(), columns=["Cities (10)"])
        st.dataframe(citiesdf)
    with col22:
        st.write("The data includes the following columns:")
        coldf = pd.DataFrame(df.columns, columns=["Columns (26)"])
        st.dataframe(coldf, height=388)

elif selected == "Programming information":
    tab1, tab2 = st.tabs(["Development Proces", "Libraries Used"])

elif selected == "Prediction Panel":

    st.title("Prediction Panel")
    st.header("Select the properties you want to predict")
    col01, col02, col03 = st.columns([0.4, 0.4, 2])
    with col01:
        st.selectbox("Select property type", df["Property_Type"].unique())
        st.slider("Select land area", 0, df["Land_Area"].max() * 2)
        st.slider("Select floor area", 0, df["Floor_Area"].max() * 3)
        st.selectbox("Select condition", df["Condition"].unique())
        st.selectbox("Select view", df["View"].unique())
        st.selectbox("Select amenities", df["Amenities"].unique())
    with col02:
        st.selectbox("Select furnishing type", df["Furnishing_Status"].unique())
