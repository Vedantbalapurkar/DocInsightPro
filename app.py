# # Q&A Chatbot
# #from langchain.llms import OpenAI

# from dotenv import load_dotenv

# load_dotenv()  # take environment variables from .env.

# import streamlit as st
# import os
# import pathlib
# import textwrap
# from PIL import Image


# import google.generativeai as genai


# #os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key="AIzaSyBlFlz8ZWICrubpTjO1dTTEVzEqI5Kv4AE")

# ## Function to load OpenAI model and get respones

# def get_gemini_response(input,image,prompt):
#     model = genai.GenerativeModel('gemini-pro-vision')
#     response = model.generate_content([input,image[0],prompt])
#     return response.text
    

# def input_image_setup(uploaded_file):
#     # Check if a file has been uploaded
#     if uploaded_file is not None:
#         # Read the file into bytes
#         bytes_data = uploaded_file.getvalue()

#         image_parts = [
#             {
#                 "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
#                 "data": bytes_data
#             }
#         ]
#         return image_parts
#     else:
#         raise FileNotFoundError("No file uploaded")


# ##initialize our streamlit app

# st.set_page_config(page_title="DocInsight Pro")

# st.header("DocInsight Pro")
# input=st.text_input("Input Prompt: ",key="input")
# uploaded_file = st.file_uploader("Choose an image...", type=None)
# image=""   
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Document.", use_column_width=True)


# submit=st.button("Tell me about the Document")

# input_prompt = """
#                You are an expert in understanding any documents.
#                You will receive input images as invoices &
#                you will have to answer questions based on the input image
#                """

# ## If ask button is clicked

# if submit:
#     image_data = input_image_setup(uploaded_file)
#     response=get_gemini_response(input_prompt,image_data,input)
#     st.subheader("The Response is")
#     st.write(response)






















from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import pandas as pd
from io import StringIO

import google.generativeai as genai

# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyBlFlz8ZWICrubpTjO1dTTEVzEqI5Kv4AE")

# Function to load OpenAI model and get responses
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# initialize our streamlit app
st.set_page_config(page_title="DocInsight Pro")

st.header("DocInsight Pro")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=None)
image = ""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document.", use_column_width=True)

submit = st.button("Tell me about the Document")

input_prompt = """
               You are an expert in understanding any documents.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """

# If ask button is clicked
if submit:
    image_data = input_image_setup(uploaded_file)
    
    # Apply the file upload handling code here
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    string_data = stringio.read()
    st.write(string_data)

    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is")
    st.write(response)
