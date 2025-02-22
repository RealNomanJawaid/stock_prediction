import streamlit as st
from scrapping_news_detection import predictionBasedOnNews
from stock_prediction import predictionBasedOnData

def main():
    # Customizing the sidebar style
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content .block-container {
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Sidebar navigation with customized radio buttons
    st.sidebar.title("IntelliTrader")
    page = st.sidebar.radio(
        "Select",
        ["**Data Prediction**", "**News Prediction**"],
        index=0,  # Default index 0 for Page 1
    )

    # Show selected page
    if page == "**Data Prediction**":
        predictionBasedOnData()
    elif page == "**News Prediction**":
        predictionBasedOnNews()   

if __name__ == "__main__":
    main()
    