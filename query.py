import os
import logging
import datetime
from typing import List, Dict, Any, Optional
import argparse

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Cấu hình logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Thiết lập logging với định dạng và nơi lưu trữ"""
    # Tạo thư mục logs nếu cần
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Định dạng log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Thiết lập root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Xóa các handler hiện có để tránh duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Thêm handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Thêm handler cho file nếu được chỉ định
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Cấu hình
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tạo log file với timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/rag_query_{timestamp}.log"

# Thiết lập logging
logger = setup_logging(log_level=logging.INFO, log_file=log_file)

def connect_to_milvus_collection(collection_name, embedding_model=None):
    """Kết nối đến collection trong Milvus."""
    logger.info(f"Đang kết nối đến collection {collection_name} trong Milvus")
    
    # Tạo embedding model nếu không được cung cấp
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("Đã khởi tạo mô hình embeddings")
    
    try:
        # Kết nối đến Milvus server
        from pymilvus import connections
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info(f"Đã kết nối thành công đến Milvus server: {MILVUS_HOST}:{MILVUS_PORT}")
        
        # Kiểm tra xem collection có tồn tại không
        from pymilvus import utility
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' không tồn tại")
            return None
        
        # Kết nối đến collection
        vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=collection_name,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
        )
        
        logger.info(f"Đã kết nối thành công đến collection {collection_name}")
        return vector_store
    except Exception as e:
        logger.error(f"Lỗi khi kết nối đến Milvus: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def setup_rag_chain(vector_store, model_name="gpt-4o-mini", temperature=0):
    """Thiết lập chuỗi RAG với LangChain."""
    logger.info(f"Đang thiết lập chuỗi RAG với mô hình {model_name}")
    
    # Tạo retriever từ vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Lấy 10 tài liệu liên quan nhất
    )
    
    # Tạo LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Tạo prompt template
    template = """
    Bạn là một trợ lý AI có khả năng trả lời câu hỏi dựa trên thông tin được cung cấp.
    Sử dụng ngữ cảnh dưới đây để trả lời câu hỏi một cách chính xác và đầy đủ.
    Nếu thông tin không có trong ngữ cảnh, hãy thành thật nói rằng bạn không biết.
    Không được tạo ra thông tin mà không có trong ngữ cảnh.

    Ngữ cảnh:
    {context}

    Câu hỏi: {question}
    
    Trả lời:
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Khởi tạo chuỗi RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Kết hợp tất cả tài liệu vào một ngữ cảnh
        retriever=retriever,
        return_source_documents=True,  # Trả về tài liệu nguồn
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    logger.info("Đã thiết lập chuỗi RAG thành công")
    return qa_chain

def process_query(query, qa_chain):
    """Xử lý query và trả về kết quả."""
    logger.info(f"Đang xử lý query: {query}")
    
    try:
        # Thực hiện truy vấn
        result = qa_chain({"query": query})
        
        # Tách kết quả và tài liệu nguồn
        answer = result["result"]
        source_docs = result["source_documents"]
        
        logger.info("Đã xử lý query thành công")
        
        return {
            "answer": answer,
            "source_documents": source_docs
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "answer": "Xảy ra lỗi khi xử lý câu hỏi của bạn.",
            "error": str(e)
        }

def interactive_query_mode(collection_name, model_name="gpt-3.5-turbo"):
    """Chế độ tương tác cho việc truy vấn."""
    logger.info("Khởi động chế độ truy vấn tương tác")
    
    # Khởi tạo embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Kết nối đến Milvus collection
    vector_store = connect_to_milvus_collection(collection_name, embedding_model)
    if not vector_store:
        print(f"Không thể kết nối đến collection {collection_name}")
        return
    
    # Thiết lập chuỗi RAG
    qa_chain = setup_rag_chain(vector_store, model_name)
    
    print(f"\n{'='*50}")
    print(f"RAG Query với Milvus Collection: {collection_name}")
    print(f"Model: {model_name}")
    print("Nhập 'quit' để thoát")
    print(f"{'='*50}\n")
    
    # Vòng lặp truy vấn
    while True:
        # Lấy input từ người dùng
        query = input("\nNhập câu hỏi của bạn: ")
        
        # Kiểm tra nếu người dùng muốn thoát
        if query.lower() in ["quit", "exit", "q", "thoát"]:
            print("Đang thoát chương trình...")
            break
        
        # Xử lý query trống
        if not query.strip():
            print("Câu hỏi không được để trống. Vui lòng thử lại.")
            continue
        
        # Xử lý query
        print("\nĐang xử lý câu hỏi của bạn...")
        start_time = datetime.datetime.now()
        
        result = process_query(query, qa_chain)
        
        # Tính thời gian xử lý
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Hiển thị kết quả
        print(f"\n{'='*50}")
        print("Câu trả lời:")
        print(f"{result['answer']}")
        print(f"\nThời gian xử lý: {duration:.2f} giây")
        
        # # Hiển thị thông tin về tài liệu nguồn
        # if "source_documents" in result:
        #     print("\nNguồn tham khảo:")
        #     for i, doc in enumerate(result["source_documents"]):
        #         print(f"  {i+1}. {doc.page_content[:100]}..." if len(doc.page_content) > 100 else f"  {i+1}. {doc.page_content}")
        #         if hasattr(doc, 'metadata') and doc.metadata:
        #             source = doc.metadata.get("source", "Không xác định")
        #             print(f"     Nguồn: {source}")
        
        # print(f"{'='*50}")

def main():
    """Hàm chính."""
    # Tạo parser cho command line arguments
    parser = argparse.ArgumentParser(description="RAG Query với Milvus")
    parser.add_argument("--collection", "-c", type=str, default="test_collection",
                       help="Tên collection trong Milvus (mặc định là test_collection)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                       help="Tên model OpenAI (mặc định là gpt-4o-mini)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Khởi động chế độ tương tác
    interactive_query_mode(args.collection, args.model)

if __name__ == "__main__":
    main()