import os

import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    As a highly skilled doctor, your task is to assess the patient's clinical condition by asking a variety of clinical questions.
    The patient is a {gender} aged {age} years old.
    based on context, chat history and a human input, if you come to a conclusion on the medical condition of the patient,
    Summarise the patient symptoms, provide a diagnosis, treatment plan and drug prescriptions in different sections. 
    Otherwise, ask a follow-up question to gather more information. Do not ask the same questions that is in chat history\n\n
    Chat History: \n{chat_history}\n
    Context:\n {context}\n
    Human Input: \n{human_input}\n

    Question:
    """



    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["gender", "age", "context", "human_input", "chat_history"])
    memory = ConversationBufferMemory(input_key="chat_history")
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt, verbose=True, memory=memory)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "What is the medical condition that you are experiencing?"}]

def user_input(user_input):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)

    chain = get_conversational_chain()

    response = chain(
        {
            "gender": st.session_state.gender,
            "age": st.session_state.age,
            "input_documents": docs,
            "human_input": user_input,
            "chat_history": st.session_state.messages
        }
    )

    print(response)
    return response

def main():
    st.set_page_config(
        page_title="MediSync Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("MediSync Symptoms Triaging System")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.radio(
            "Gender:",
            options=["male", "female"],
            horizontal=True,
        )
        st.session_state["gender"] = gender

    with col2:
        age = st.text_input('Age:', value='35')
        st.session_state["age"] = age

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Describe the medical condition that you are experiencing"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
