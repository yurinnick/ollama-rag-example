import os
import logging

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


logger = logging.getLogger(__name__)


class Learner:
    ALLOWED_EXTENTIONS: List[str] = [".pdf"]

    def __init__(
        self,
        ollama_host: str,
        vector_collection: str,
        vector_store_path: str,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.ollama_host = ollama_host
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vector_collection = vector_collection
        self.vector_store_path = vector_store_path
        self.chunk_size = 7500
        self.chunk_overlap = 100

        embedding = OllamaEmbeddings(
            model=str(self.embedding_model),
            base_url=self.ollama_host,
        )
        self.vector_db = Chroma(
            collection_name=self.vector_collection,
            persist_directory=self.vector_store_path,
            embedding_function=embedding,
        )

    def is_supported(self, filename: os.PathLike) -> bool:
        for ext in self.ALLOWED_EXTENTIONS:
            if filename.lower().endswith(ext):
                return True
        return False

    def __load_pdf(self, filepath: os.PathLike) -> List[Document]:
        loader = UnstructuredPDFLoader(file_path=filepath)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return loader.load_and_split(text_splitter)

    def learn(self, files: List[os.PathLike]) -> None:
        for filepath in files:
            if not filepath:
                continue
            if not self.is_supported(filepath):
                continue

            chunks = self.__load_pdf(filepath)
            self.vector_db.add_documents(chunks)

    def __get_prompt(self):
        prompt_tmpl = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant. Generate five reworded versions of the user question
            to improve document retrieval. Original question: {question}""",
        )
        template = "Answer the question based ONLY on this context:\n{context}\nQuestion: {question}"
        prompt = ChatPromptTemplate.from_template(template)
        return prompt_tmpl, prompt

    def query(self, input: str):
        if input:
            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.ollama_host,
            )
            prompt_tmpl, prompt = self.__get_prompt()
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(),
                llm=llm,
                prompt=prompt_tmpl,
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            return chain.invoke(input)
        return "No input provided"
