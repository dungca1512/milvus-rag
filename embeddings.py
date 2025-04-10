import os
import logging
import datetime
from typing import List, Dict, Any, Optional
import uuid
import argparse

# Để đọc file docx
import docx2txt

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
log_file = f"logs/milvus_embedding_{timestamp}.log"

# Thiết lập logging
logger = setup_logging(log_level=logging.INFO, log_file=log_file)

def create_embeddings_model():
    """Tạo mô hình embeddings sử dụng OpenAI."""
    logger.info("Khởi tạo mô hình OpenAI Embeddings")
    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("Đã khởi tạo mô hình embeddings thành công")
        return embedding_model
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo mô hình embeddings: {e}")
        return None

def create_milvus_store(embeddings, collection_name):
    """Tạo Milvus vector store mới."""
    logger.info(f"Tạo Milvus vector store với collection name: {collection_name}")
    try:
        # Kiểm tra kết nối đến Milvus server
        from pymilvus import connections
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info(f"Đã kết nối thành công đến Milvus server: {MILVUS_HOST}:{MILVUS_PORT}")
        
        # Kiểm tra xem collection đã tồn tại chưa
        from pymilvus import utility
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' đã tồn tại và sẽ bị xóa")
            utility.drop_collection(collection_name)
            logger.info(f"Đã xóa collection '{collection_name}'")
        
        # Tạo vector store từ texts (để khởi tạo collection)
        # Thêm một văn bản mẫu để khởi tạo collection
        ids = ["init-doc"]
        vector_store = Milvus.from_texts(
            texts=["Tài liệu khởi tạo"],
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            ids=ids  # Cung cấp ID để tránh lỗi
        )
        
        logger.info(f"Đã tạo collection {collection_name} mới thành công")
        return vector_store
    except Exception as e:
        logger.error(f"Lỗi khi tạo Milvus vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def embed_and_store_documents(texts, metadatas=None, collection_name=None, chunk_size=1000, chunk_overlap=100):
    """Tạo embeddings cho văn bản và lưu vào Milvus."""
    # Đặt tên collection mặc định nếu không được cung cấp
    if not collection_name:
        collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Bắt đầu quá trình tạo embeddings và lưu trữ vào collection: {collection_name}")
    logger.info(f"Số lượng văn bản: {len(texts)}")
    
    # Kiểm tra API key
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY không được cung cấp hoặc trống")
        return {
            "status": "error",
            "error": "OPENAI_API_KEY không tồn tại"
        }
    
    # Khởi tạo mô hình embeddings
    embeddings = create_embeddings_model()
    if not embeddings:
        return {
            "status": "error",
            "error": "Không thể khởi tạo mô hình embeddings"
        }
    
    # Tạo Milvus vector store
    vector_store = create_milvus_store(embeddings, collection_name)
    if not vector_store:
        return {
            "status": "error",
            "error": "Không thể tạo Milvus vector store"
        }
    
    # Tạo danh sách Document
    logger.info("Tạo danh sách Document từ văn bản đầu vào")
    documents = []
    if metadatas:
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            documents.append(Document(page_content=text, metadata=metadata))
            # Log mỗi 100 tài liệu để tránh log quá nhiều
            if i % 100 == 0 and i > 0:
                logger.debug(f"Đã tạo {i} Document")
    else:
        for i, text in enumerate(texts):
            documents.append(Document(page_content=text))
            if i % 100 == 0 and i > 0:
                logger.debug(f"Đã tạo {i} Document")
    
    # Phân đoạn tài liệu nếu cần thiết
    if chunk_size > 0:
        logger.info(f"Tiến hành phân đoạn tài liệu với chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        original_doc_count = len(documents)
        documents = text_splitter.split_documents(documents)
        logger.info(f"Đã phân tài liệu từ {original_doc_count} thành {len(documents)} đoạn")
    
    # Lưu tài liệu vào Milvus
    try:
        logger.info(f"Bắt đầu tạo embeddings và lưu {len(documents)} tài liệu vào Milvus")
        
        # Tính toán kích thước trung bình của tài liệu để log
        avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
        logger.info(f"Kích thước trung bình của mỗi tài liệu: {avg_doc_length:.2f} ký tự")
        
        # Thời gian bắt đầu để tính thời gian thực hiện
        start_time = datetime.datetime.now()
        
        # Tạo IDs cho từng tài liệu
        doc_ids = [f"doc-{uuid.uuid4().hex[:16]}" for _ in range(len(documents))]
        
        # Tách các trường cần thiết
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Sử dụng add_texts với ids thay vì add_documents
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=doc_ids)
        
        # Tính thời gian thực hiện
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Đã lưu {len(documents)} tài liệu vào Milvus collection '{collection_name}' trong {duration:.2f} giây")
        logger.info(f"Tốc độ xử lý: {len(documents)/duration:.2f} tài liệu/giây")
        
        return {
            "collection_name": collection_name,
            "document_count": len(documents),
            "duration_seconds": duration,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Lỗi khi lưu tài liệu vào Milvus: {e}")
        # Log chi tiết hơn về lỗi để debug
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "collection_name": collection_name,
            "status": "error",
            "error": str(e)
        }

def process_docx_file(file_path, collection_name=None, chunk_size=1000, chunk_overlap=100):
    """Đọc file DOCX, xử lý và lưu vào Milvus."""
    logger.info(f"Bắt đầu xử lý file DOCX: {file_path}")
    
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            logger.error(f"File không tồn tại: {file_path}")
            return {
                "status": "error",
                "error": f"File không tồn tại: {file_path}"
            }
        
        # Kiểm tra đuôi file có phải .docx không
        if not file_path.lower().endswith('.docx'):
            logger.error(f"File không phải định dạng DOCX: {file_path}")
            return {
                "status": "error",
                "error": f"File phải có định dạng .docx: {file_path}"
            }
        
        # Đọc nội dung file DOCX
        logger.info(f"Đọc nội dung file DOCX: {file_path}")
        file_size = os.path.getsize(file_path) / 1024  # kích thước file theo KB
        logger.info(f"Kích thước file: {file_size:.2f} KB")
        
        # Sử dụng docx2txt để trích xuất văn bản
        content = docx2txt.process(file_path)
        
        logger.info(f"Đã đọc nội dung file DOCX thành công. Độ dài nội dung: {len(content)} ký tự")
        
        # Tạo metadata từ thông tin file
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        
        metadata = {
            "source": file_path,
            "file_name": file_name,
            "file_extension": file_ext,
            "file_size_kb": file_size,
            "modified_time": modified_time.isoformat(),
            "processed_time": datetime.datetime.now().isoformat(),
            "file_type": "docx"
        }
        
        logger.info(f"Đã tạo metadata cho file DOCX: {metadata}")
        
        # Gọi hàm để tạo embeddings và lưu
        if not collection_name:
            collection_name = f"docx_{os.path.splitext(file_name)[0]}_{uuid.uuid4().hex[:6]}"
            
        logger.info(f"Bắt đầu tạo embedding với collection name: {collection_name}")
        
        # Gọi hàm để tạo embeddings và lưu
        result = embed_and_store_documents(
            texts=[content], 
            metadatas=[metadata],
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Kết quả xử lý file DOCX: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file DOCX {file_path}: {e}")
        # Log stack trace để debug
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Hàm chính để xử lý file DOCX và lưu vào Milvus."""
    # Tạo parser cho command line arguments
    parser = argparse.ArgumentParser(description="Xử lý file DOCX và lưu embeddings vào Milvus")
    parser.add_argument("--file", "-f", type=str, required=True,
                       help="Đường dẫn đến file DOCX cần xử lý")
    parser.add_argument("--collection", "-c", type=str, default=None,
                       help="Tên collection trong Milvus (mặc định: tạo từ tên file)")
    parser.add_argument("--chunk-size", "-s", type=int, default=1000,
                       help="Kích thước mỗi đoạn văn bản (mặc định: 1000)")
    parser.add_argument("--chunk-overlap", "-o", type=int, default=100,
                       help="Độ chồng lấp giữa các đoạn (mặc định: 100)")
    
    # Parse arguments
    args = parser.parse_args()
    
    logger.info("=== BẮT ĐẦU CHƯƠNG TRÌNH ===")
    logger.info(f"File DOCX: {args.file}")
    logger.info(f"Collection: {args.collection if args.collection else 'Auto-generated'}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    
    # Xử lý file DOCX
    result = process_docx_file(
        file_path=args.file,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Hiển thị kết quả
    if result["status"] == "success":
        print(f"\n{'='*50}")
        print(f"XỬ LÝ FILE DOCX THÀNH CÔNG")
        print(f"Collection: {result['collection_name']}")
        print(f"Số lượng đoạn: {result['document_count']}")
        print(f"Thời gian xử lý: {result['duration_seconds']:.2f} giây")
        print(f"{'='*50}")
        print("\nBạn có thể sử dụng collection này để truy vấn bằng lệnh:")
        print(f"python rag_query.py --collection {result['collection_name']}")
    else:
        print(f"\n{'='*50}")
        print(f"XỬ LÝ FILE DOCX THẤT BẠI")
        print(f"Lỗi: {result.get('error', 'Không xác định')}")
        print(f"{'='*50}")
    
    logger.info("=== KẾT THÚC CHƯƠNG TRÌNH ===")

if __name__ == "__main__":
    main()