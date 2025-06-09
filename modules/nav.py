import streamlit as st


def Navbar():
    with st.sidebar:
        st.page_link('streamlit_app.py', label='Intro', icon='🔥')
        st.page_link('pages/1_Data_Analysis.py', label='Data Analysis', icon='📊')
        st.page_link('pages/2_Compare_Models.py', label='ML Models', icon='⚙️')
        st.page_link('pages/3_Classify_Entry.py', label='Classify Custom Entry', icon='🏷️')
        st.page_link('pages/4_Deep_Learning.py', label='Deep Learning Models', icon='🧠')