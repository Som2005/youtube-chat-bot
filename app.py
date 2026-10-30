from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_huggingface import HuggingFaceEndpointEmbeddings , HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
import streamlit as st
load_dotenv()
st.header("Youtube video chatbot")
st.subheader("paste youtube link")

user_input = st.text_input("Enter YouTube URL")

if user_input.strip():
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=user_input,
        language=["hi"],
        add_video_info=False
    )
    docs=loader.load()
#    print(len(docs))
#   print(docs[0])
    splitrer=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs_n=splitrer.split_documents(docs)
# print(docs_n[0])
# print(len(docs_n))
    embed=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vectorstore=FAISS.from_documents(
        documents=docs_n,
        embedding=embed
)
# print(vectorstore.index_to_docstore_id)
# doc_id=list(vectorstore.index_to_docstore_id.values())[0]
# ids=vectorstore.docstore.search(doc_id)
# print(ids.page_content)

    retrievers=vectorstore.as_retriever(
        search_type="similarity",
        search_kwrags={"k":2}
)
# print(retrievers.invoke("what is that in video?"))

    prompt=PromptTemplate(
        template="""
        you are a assistant.
        answer only from provided transcipt context.
        if the context is insufficient, just say you don`t know.
        {context}
        question:{question} 
    """,
    input_variables=["context","question"]
)

    model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# question="in this video what is experiment?"
# retrival__docs=retrievers.invoke(question)

    def context(r_docs):
        contexts="\n\n".join(doc.page_content for doc in r_docs)
        return contexts

    parallel_chain=RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retrievers | RunnableLambda(context)
})

    parser=StrOutputParser()
    question_input=st.text_input("ask your doubt in this video")
    if st.button:
        main_Chain=parallel_chain | prompt | model | parser
        result=main_Chain.invoke({
            "question":question_input
            })
        st.write(result)
elif TranscriptsDisabled:
    st.write("in this video transscript not found")