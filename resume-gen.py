import os
import time
import langchain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import pytesseract as pyt
import numpy as np
from langchain.messages import SystemMessage , HumanMessage
from langchain.agents import create_agent
import streamlit as st


st.title("AI RESUME GENERATOR")

TAVILY_KEY = st.sidebar.text_input("Tavily API Key" , type = "password")
GROQ_KEY = st.sidebar.text_input("Groq API Key" , type = "password")
GEMINI_KEY = st.sidebar.text_input("Google API Key" , type = "password")

if not GEMINI_KEY :
  st.warning("Provide Google API Key")


# tool 1
from tavily import TavilyClient
client = TavilyClient(TAVILY_KEY)
def news(query) :
    """this function helps to get latest news
    or latest jobs related to
    the given query using tavily"""

    return client.search(
        query=query,
        topic="news",
        max_results=5
    )["results"]

# step 1 : Model Creation
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GEMINI_KEY
)

model2 = ChatGroq(
    model = "qwen/qwen3.6-27b",
    api_key = GROQ_KEY
)

agent = create_agent(
    model = model,
    tools = [news]
)

# lets generate prompt for llm for resume building

def prompt_generator():
 prompt = """You are a helpful ai resume maker,
 i want you to use chain_of_thoughts and give detailed prompt for model
 to where user want to generate resume for
 freshers or expirienced one in the html format ,
 you have to give prompt set of instructions and
 make sure to keep design professsional"""

 response = model.invoke(prompt)
 prompt = response.content[0]['text']
 filename = "prompt.txt"
 with open(filename, 'w') as f:
  f.write(prompt)

prompt_generator()


def prompt_reader() :
  with open("prompt.txt","r") as f:
    prompt = f.read()
  return prompt

prompt = """ I want complete professional resume with dynamic
Design using advance css and js and must show user input details
System instuctions : Give only html code as output"""

main_prompt = prompt + prompt_reader()

user_info = st.text_input("Give your info")
user_pic = st.sidebar.file_uploader("Upload photo",type = "image/jpeg")

user_prompt = f"""
  user details = {user_info}
  use profile pic from given image {user_pic}
"""

final_prompt = main_prompt + user_prompt

if st.button("Generate Resume") : 
  with st.spinner("Agent creating resume...") :
    response = agent.invoke({'messages' : [{'role' : 'user' , 'content' : final_prompt}]})
    code = response['messages'][-1].content[-1]['text']
    st.html(code, width="stretch", unsafe_allow_javascript=True)







