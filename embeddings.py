import os
import logging
import datetime
import uuid
import argparse
import json
from typing import List, Dict, Any, Optional
import time

# Để đọc file docx
import docx2txt

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Import từ module chunking.py (nếu có sẵn, nếu không sẽ tự xử lý)
try:
    from chunking import setup_logging, split_documents, detect_document_structure, chunk_document
    chunking_module_available = True
except ImportError:
    chunking_module_available = False
    # Sẽ tự xử lý chunking nếu cần

# Cấu hình
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tạo log file với timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/milvus_embedding_{timestamp}.log"

# Thiết lập logging
if not chunking_module_available:
    # Định nghĩa hàm logging nếu không import được từ chunking.py
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

def embed_chunks(chunks, collection_name=None, log_dir=None):
    """
    Tạo embeddings cho các chunk và lưu vào Milvus.
    Ghi log quá trình embedding cho từng chunk.
    
    Args:
        chunks (List[Document]): Danh sách các chunk cần tạo embedding
        collection_name (str, optional): Tên collection trong Milvus
        log_dir (str, optional): Thư mục để lưu log chi tiết về embeddings
        
    Returns:
        Dict: Kết quả của quá trình embedding
    """
    # Đặt tên collection mặc định nếu không được cung cấp
    if not collection_name:
        collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    
    print(f"Bắt đầu quá trình tạo embeddings và lưu trữ vào collection: {collection_name}")
    logger.info(f"Bắt đầu quá trình tạo embeddings và lưu trữ vào collection: {collection_name}")
    logger.info(f"Số lượng chunks: {len(chunks)}")
    print(f"Số lượng chunks: {len(chunks)}")
    
    # Kiểm tra API key
    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY không được cung cấp hoặc trống"
        logger.error(error_msg)
        print(f"Lỗi: {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        }
    
    # Khởi tạo mô hình embeddings
    embeddings = create_embeddings_model()
    if not embeddings:
        error_msg = "Không thể khởi tạo mô hình embeddings"
        print(f"Lỗi: {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        }
    
    # Tạo Milvus vector store
    vector_store = create_milvus_store(embeddings, collection_name)
    if not vector_store:
        error_msg = "Không thể tạo Milvus vector store"
        print(f"Lỗi: {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        }
    
    # Tạo thư mục log nếu được chỉ định
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Đã tạo thư mục log: {log_dir}")
    
    # Ghi log chi tiết
    embedding_log_file = None
    if log_dir:
        embedding_log_file = os.path.join(log_dir, f"embedding_details_{collection_name}.jsonl")
        logger.info(f"File log chi tiết: {embedding_log_file}")
    
    try:
        logger.info(f"Bắt đầu tạo embeddings và lưu {len(chunks)} chunks vào Milvus")
        print(f"Bắt đầu tạo embeddings và lưu {len(chunks)} chunks vào Milvus")
        
        # Tính toán kích thước trung bình của chunk để log
        avg_chunk_length = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        logger.info(f"Kích thước trung bình của mỗi chunk: {avg_chunk_length:.2f} ký tự")
        print(f"Kích thước trung bình của mỗi chunk: {avg_chunk_length:.2f} ký tự")
        
        # Thời gian bắt đầu để tính thời gian thực hiện
        start_time = datetime.datetime.now()
        
        # Xử lý từng chunk và tạo embedding
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            
            # Tạo ID cho chunk
            chunk_id = f"doc-{uuid.uuid4().hex[:16]}"
            
            # Log thông tin
            logger.info(f"Đang xử lý chunk #{i+1}/{len(chunks)}: {len(chunk.page_content)} ký tự")
            
            # Trích xuất dữ liệu chunk
            chunk_text = chunk.page_content
            chunk_metadata = chunk.metadata
            
            try:
                # In thông tin về nội dung chunk (100 ký tự đầu)
                chunk_preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\n{'='*80}\nEmbedding chunk #{i+1}/{len(chunks)}:\n{'-'*80}\n{chunk_preview}\n{'='*80}")
                
                # Tạo embedding và lưu vào Milvus
                vector_store.add_texts(
                    texts=[chunk_text],
                    metadatas=[chunk_metadata],
                    ids=[chunk_id]
                )
                
                # Tính thời gian thực hiện
                chunk_duration = time.time() - chunk_start_time
                
                # In thông tin về chunk đã xử lý
                logger.info(f"Đã tạo embedding cho chunk #{i+1}: {chunk_duration:.2f} giây")
                print(f"Đã tạo embedding: {chunk_duration:.2f} giây\n")
                
                # Ghi log chi tiết nếu có file log
                if embedding_log_file:
                    with open(embedding_log_file, 'a', encoding='utf-8') as f:
                        log_entry = {
                            "chunk_id": chunk_id,
                            "chunk_number": i + 1,
                            "chunk_size": len(chunk_text),
                            "chunk_first_100_chars": chunk_text[:100] + ("..." if len(chunk_text) > 100 else ""),
                            "processing_time_seconds": chunk_duration,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "metadata": {k: str(v) for k, v in chunk_metadata.items()}  # Convert all values to strings for JSON
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo embedding cho chunk #{i+1}: {e}")
                print(f"Lỗi khi tạo embedding: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Tính thời gian thực hiện tổng cộng
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Đã lưu {len(chunks)} chunks vào Milvus collection '{collection_name}' trong {duration:.2f} giây")
        logger.info(f"Tốc độ xử lý: {len(chunks)/duration:.2f} chunks/giây")
        
        print(f"\n{'='*80}")
        print(f"Đã lưu {len(chunks)} chunks vào Milvus collection '{collection_name}' trong {duration:.2f} giây")
        print(f"Tốc độ xử lý: {len(chunks)/duration:.2f} chunks/giây")
        print(f"{'='*80}")
        
        return {
            "collection_name": collection_name,
            "document_count": len(chunks),
            "duration_seconds": duration,
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Lỗi khi lưu chunks vào Milvus: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        print(f"Lỗi: {error_msg}")
        
        return {
            "collection_name": collection_name,
            "status": "error",
            "error": str(e)
        }

def process_file_embeddings(file_path, collection_name=None, chunk_size=1000, chunk_overlap=100, split_method=None, log_dir=None):
    """
    Xử lý file, phân chia thành chunks và tạo embeddings.
    
    Args:
        file_path (str): Đường dẫn đến file cần xử lý
        collection_name (str, optional): Tên collection trong Milvus
        chunk_size (int): Kích thước tối đa mỗi đoạn
        chunk_overlap (int): Độ chồng lấp giữa các đoạn
        split_method (str, optional): Phương pháp phân chia
        log_dir (str, optional): Thư mục để lưu log chi tiết
        
    Returns:
        Dict: Kết quả xử lý
    """
    print(f"Bắt đầu xử lý file: {file_path}")
    logger.info(f"Bắt đầu xử lý file: {file_path}")
    
    # Tạo collection_name nếu không được cung cấp
    if not collection_name:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        collection_name = f"{base_name}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Tự động tạo tên collection: {collection_name}")
        print(f"Tự động tạo tên collection: {collection_name}")
    
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            error_msg = f"File không tồn tại: {file_path}"
            logger.error(error_msg)
            print(f"Lỗi: {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
        
        # Phân chia file thành các chunks
        if chunking_module_available:
            print(f"Đang phân chia file thành các chunks...")
            logger.info(f"Sử dụng module chunking.py để phân chia file")
            chunks, used_split_method = chunk_document(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                split_method=split_method
            )
            
            if not chunks:
                error_msg = "Không thể phân chia file thành chunks"
                logger.error(error_msg)
                print(f"Lỗi: {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg
                }
            
            logger.info(f"Đã phân chia file thành {len(chunks)} chunks với phương pháp {used_split_method}")
            print(f"Đã phân chia file thành {len(chunks)} chunks với phương pháp {used_split_method}")
        else:
            # Đọc nội dung file nếu không có module chunking.py
            logger.info(f"Module chunking.py không khả dụng, tự xử lý chunking")
            print(f"Module chunking.py không khả dụng, tự xử lý file...")
            
            # Đọc nội dung file
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.docx':
                logger.info(f"Đọc file DOCX: {file_path}")
                content = docx2txt.process(file_path)
            elif file_ext == '.txt':
                logger.info(f"Đọc file TXT: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                error_msg = f"Định dạng file không được hỗ trợ: {file_ext}"
                logger.error(error_msg)
                print(f"Lỗi: {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg
                }
            
            # Tạo metadata
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # kích thước file theo KB
            modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "file_extension": file_ext,
                "file_size_kb": file_size,
                "modified_time": modified_time.isoformat(),
                "processed_time": datetime.datetime.now().isoformat()
            }
            
            # Sử dụng mặc định RecursiveCharacterTextSplitter
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Tạo một document và phân chia
            document = Document(page_content=content, metadata=metadata)
            chunks = text_splitter.split_documents([document])
            used_split_method = "semantic"
            
            logger.info(f"Đã phân chia file thành {len(chunks)} chunks với phương pháp basic")
            print(f"Đã phân chia file thành {len(chunks)} chunks")
            
        # Tạo embeddings và lưu vào Milvus
        result = embed_chunks(
            chunks=chunks,
            collection_name=collection_name,
            log_dir=log_dir
        )
        
        # Thêm thông tin về phương pháp chunking vào kết quả
        if result["status"] == "success":
            result["chunking_method"] = used_split_method
        
        return result
        
    except Exception as e:
        error_msg = f"Lỗi khi xử lý file {file_path}: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        print(f"Lỗi: {error_msg}")
        
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Hàm chính để xử lý file và tạo embeddings."""
    # Tạo parser cho command line arguments
    parser = argparse.ArgumentParser(description="Xử lý file và tạo embeddings lưu vào Milvus")
    parser.add_argument("--file", "-f", type=str, required=True,
                       help="Đường dẫn đến file cần xử lý")
    parser.add_argument("--collection", "-c", type=str, default=None,
                       help="Tên collection trong Milvus (mặc định: tạo từ tên file)")
    parser.add_argument("--chunk-size", "-s", type=int, default=1000,
                       help="Kích thước tối đa mỗi đoạn văn bản (mặc định: 1000)")
    parser.add_argument("--chunk-overlap", "-o", type=int, default=100,
                       help="Độ chồng lấp giữa các đoạn (mặc định: 100)")
    parser.add_argument("--split-method", "-m", type=str, default=None, 
                       choices=["paragraph", "markdown", "semantic", "semantic_embedding", "token"],
                       help="Phương pháp phân đoạn (mặc định: tự động phát hiện)")
    parser.add_argument("--log-dir", "-l", type=str, default="logs/embeddings",
                       help="Thư mục để lưu log chi tiết (mặc định: logs/embeddings)")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Milvus server host (mặc định: localhost)")
    parser.add_argument("--port", type=str, default="19530",
                       help="Milvus server port (mặc định: 19530)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Cập nhật cấu hình Milvus
    global MILVUS_HOST, MILVUS_PORT
    MILVUS_HOST = args.host
    MILVUS_PORT = args.port
    
    # Hiển thị thông tin cấu hình
    print(f"\n{'='*80}")
    print(f"BẮT ĐẦU TẠO EMBEDDINGS")
    print(f"{'='*80}")
    print(f"File: {args.file}")
    print(f"Collection: {args.collection if args.collection else 'Auto-generated'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Phương pháp phân đoạn: {args.split_method if args.split_method else 'Tự động phát hiện'}")
    print(f"Milvus server: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"{'='*80}\n")
    
    logger.info("=== BẮT ĐẦU TẠO EMBEDDINGS ===")
    logger.info(f"File: {args.file}")
    logger.info(f"Collection: {args.collection if args.collection else 'Auto-generated'}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    logger.info(f"Phương pháp phân đoạn: {args.split_method if args.split_method else 'Tự động phát hiện'}")
    logger.info(f"Thư mục log: {args.log_dir}")
    logger.info(f"Milvus server: {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Kiểm tra API key
    if not OPENAI_API_KEY:
        print("Lỗi: Biến môi trường OPENAI_API_KEY chưa được thiết lập.")
        print("Vui lòng thiết lập biến môi trường OPENAI_API_KEY trước khi chạy chương trình.")
        logger.error("OPENAI_API_KEY không được thiết lập")
        return
    
    # Xử lý file và tạo embeddings
    result = process_file_embeddings(
        file_path=args.file,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split_method=args.split_method,
        log_dir=args.log_dir
    )
    
    # Hiển thị kết quả
    if result["status"] == "success":
        print(f"\n{'='*80}")
        print(f"TẠO EMBEDDINGS THÀNH CÔNG")
        print(f"Collection: {result['collection_name']}")
        print(f"Số lượng chunks: {result['document_count']}")
        if "chunking_method" in result:
            print(f"Phương pháp phân đoạn: {result['chunking_method']}")
        print(f"Thời gian xử lý: {result['duration_seconds']:.2f} giây")
        print(f"{'='*80}")
        print("\nBạn có thể sử dụng collection này để truy vấn bằng lệnh:")
        print(f"python rag_query.py --collection {result['collection_name']}")
    else:
        print(f"\n{'='*80}")
        print(f"TẠO EMBEDDINGS THẤT BẠI")
        print(f"Lỗi: {result.get('error', 'Không xác định')}")
        print(f"{'='*80}")
    
    logger.info("=== KẾT THÚC TẠO EMBEDDINGS ===")

if __name__ == "__main__":
    main()