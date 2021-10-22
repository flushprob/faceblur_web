from os import P_DETACH


try:
    import streamlit as st
    import os
    import sys
    import pandas as pd
    from io import BytesIO, StringIO
    print("All module loaded")
except Exception as e:
    print( "Some Modules are Missing : {} ".format(e))

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

def main():
    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()
    
    if not file:
        show_file.info("Please Upload a file : {} ".format(' '.join(["png","jpg"])))
        return
    content = file.getvalue()

    if isinstance(file, BytesIO):
        show_file.image(file)
    else:
        df = pd.read_csv(file)
        st.dataframe(df.head(2))
        file.close()


main()

## Title
st.title('Streamlit Tutorial')
## Header/Subheader
st.header('This is header')
st.subheader('This is subheader')
## Text
st.text("Hello Streamlit! 이 글은 튜토리얼 입니다.")