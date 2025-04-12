import os
import logging
import datetime
import argparse
import re
from typing import List, Dict, Any, Optional
import docx2txt
from langchain.schema import Document

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

# Tạo log file với timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/chunking_{timestamp}.log"

# Thiết lập logging
logger = setup_logging(log_level=logging.INFO, log_file=log_file)

class SectionSplitter:
    """
    Phân chia tài liệu theo các đầu mục lớn (A, B, C) và các chuyên môn trong C.
    """
    
    def __init__(self):
        # Các mẫu để nhận diện các đầu mục lớn
        self.main_section_patterns = [
            r'^\s*[A-C]\s*[-–—]\s*(.+)\s*$',       # A - NĂNG LỰC CỐT LÕI
            r'^\s*[A-C]\s*\.\s*(.+)\s*$',          # A. NĂNG LỰC CỐT LÕI
            r'^\s*[A-C]\s+(.+)\s*$',               # A NĂNG LỰC CỐT LÕI
        ]
        
        # Mẫu để nhận diện các chuyên môn trong mục C
        self.specialty_patterns = [
            r'^\s*C\s*[-–—]\s*NĂNG LỰC CHUYÊN MÔN:\s*(.+)\s*$',  # C - NĂNG LỰC CHUYÊN MÔN: ANDROID DEVELOPER
            r'^\s*C\s*\.\s*NĂNG LỰC CHUYÊN MÔN:\s*(.+)\s*$',      # C. NĂNG LỰC CHUYÊN MÔN: ANDROID DEVELOPER
            r'^\s*NĂNG LỰC CHUYÊN MÔN:\s*(.+)\s*$',              # NĂNG LỰC CHUYÊN MÔN: ANDROID DEVELOPER
        ]
    
    def identify_main_sections(self, text):
        """
        Nhận diện các đầu mục lớn A, B, C trong văn bản.
        Returns: List of (start_pos, end_pos, section_name, section_content)
        """
        lines = text.split('\n')
        sections = []
        current_position = 0
        section_start = -1
        current_section = ""
        
        for i, line in enumerate(lines):
            line_start_pos = current_position
            current_position += len(line) + 1  # +1 cho ký tự xuống dòng
            
            # Kiểm tra xem dòng hiện tại có phải là đầu mục lớn không
            is_main_section = False
            section_name = ""
            
            for pattern in self.main_section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    is_main_section = True
                    section_name = line.strip()
                    break
            
            # Kiểm tra xem có phải đầu mục chuyên môn trong C không
            if not is_main_section and i > 0:
                for pattern in self.specialty_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        is_main_section = True
                        section_name = line.strip()
                        break
            
            # Nếu tìm thấy đầu mục mới
            if is_main_section:
                # Nếu đã có đầu mục trước đó, lưu lại
                if section_start >= 0 and current_section:
                    section_content = "\n".join(lines[section_start:i])
                    sections.append((section_start, line_start_pos, current_section, section_content))
                
                # Bắt đầu đầu mục mới
                section_start = i
                current_section = section_name
        
        # Thêm đầu mục cuối cùng nếu có
        if section_start >= 0 and current_section:
            section_content = "\n".join(lines[section_start:])
            sections.append((section_start, current_position, current_section, section_content))
        
        return sections
    
    def extract_section_type(self, section_name):
        """
        Trích xuất loại đầu mục (A, B, C) và chuyên môn từ tên đầu mục.
        """
        # Kiểm tra xem có phải là đầu mục A, B không
        for letter in ["A", "B"]:
            if section_name.startswith(letter):
                return letter, None
        
        # Kiểm tra xem có phải là đầu mục C với chuyên môn không
        for pattern in self.specialty_patterns:
            match = re.match(pattern, section_name)
            if match:
                specialty = match.group(1).strip()
                return "C", specialty
        
        # Nếu bắt đầu bằng C nhưng không có chuyên môn
        if section_name.startswith("C"):
            return "C", None
        
        # Mặc định
        return None, None
    
    def split_documents(self, documents):
        """
        Phân chia documents thành các chunk theo các đầu mục A, B, C và chuyên môn.
        """
        result = []
        
        for doc in documents:
            text = doc.page_content
            base_metadata = doc.metadata
            
            # Trích xuất các đầu mục từ tài liệu
            sections = self.identify_main_sections(text)
            
            if not sections:
                # Nếu không tìm thấy đầu mục nào, giữ nguyên document
                result.append(doc)
                continue
            
            # Tạo document cho từng đầu mục
            for _, _, section_name, section_content in sections:
                # Trích xuất loại đầu mục và chuyên môn
                section_type, specialty = self.extract_section_type(section_name)
                
                # Tạo metadata cho đầu mục
                section_metadata = base_metadata.copy()
                section_metadata["section_name"] = section_name
                section_metadata["section_type"] = section_type
                
                if specialty:
                    section_metadata["specialty"] = specialty
                
                # Tạo document cho đầu mục
                section_doc = Document(
                    page_content=section_content,
                    metadata=section_metadata
                )
                result.append(section_doc)
        
        return result

def split_documents(documents):
    """
    Phân chia danh sách document theo các đầu mục A, B, C và chuyên môn.
    
    Args:
        documents: Danh sách document cần phân chia
        
    Returns:
        List[Document]: Danh sách document đã được phân chia
        str: Phương pháp phân chia đã sử dụng
    """
    if not documents:
        return [], "none"
    
    # Sử dụng SectionSplitter để phân chia theo đầu mục
    splitter = SectionSplitter()
    method = "section_based"
    
    logger.info(f"Tiến hành phân đoạn tài liệu theo phương pháp '{method}'")
    
    # Phân chia tài liệu
    original_doc_count = len(documents)
    chunked_documents = splitter.split_documents(documents)
    
    logger.info(f"Đã phân tài liệu từ {original_doc_count} thành {len(chunked_documents)} đoạn")
    
    return chunked_documents, method

def analyze_chunks(chunks):
    """
    Phân tích các chunk để hiển thị thông tin thống kê.
    """
    if not chunks:
        return {
            "count": 0,
            "avg_size": 0,
            "min_size": 0,
            "max_size": 0,
            "sections": {},
            "specialties": {}
        }
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    
    # Đếm số lượng chunk theo đầu mục
    section_counts = {"A": 0, "B": 0, "C": 0, "Khác": 0}
    specialty_counts = {}
    
    for chunk in chunks:
        section_type = chunk.metadata.get("section_type")
        if section_type in section_counts:
            section_counts[section_type] += 1
        else:
            section_counts["Khác"] += 1
        
        specialty = chunk.metadata.get("specialty")
        if specialty:
            if specialty in specialty_counts:
                specialty_counts[specialty] += 1
            else:
                specialty_counts[specialty] = 1
    
    return {
        "count": len(chunks),
        "avg_size": avg_size,
        "min_size": min_size,
        "max_size": max_size,
        "sections": section_counts,
        "specialties": specialty_counts
    }

def chunk_document(file_path, output_file=None):
    """
    Đọc và phân chia tài liệu thành các đoạn theo đầu mục A, B, C và chuyên môn.
    In ra từng đoạn chunk được tạo ra.
    
    Args:
        file_path: Đường dẫn đến tài liệu cần xử lý
        output_file: Đường dẫn đến file để lưu kết quả chunk
        
    Returns:
        List[Document]: Danh sách các đoạn đã phân chia
        str: Phương pháp phân đoạn đã sử dụng
        Dict: Thông tin thống kê về các chunk
    """
    logger.info(f"Bắt đầu phân chia tài liệu: {file_path}")
    
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            logger.error(f"File không tồn tại: {file_path}")
            return None, None, None
        
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
            logger.error(f"Định dạng file không được hỗ trợ: {file_ext}")
            return None, None, None
        
        logger.info(f"Đã đọc nội dung file thành công. Độ dài nội dung: {len(content)} ký tự")
        
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
        
        logger.info("Đã tạo metadata cho file")
        
        # Tạo document để phân chia
        document = Document(page_content=content, metadata=metadata)
        
        # Phân đoạn tài liệu
        chunks, used_split_method = split_documents(
            documents=[document]
        )
        
        # Phân tích các chunk
        chunk_stats = analyze_chunks(chunks)
        logger.info(f"Đã phân tài liệu thành {chunk_stats['count']} đoạn")
        logger.info(f"Kích thước trung bình: {chunk_stats['avg_size']:.2f} ký tự")
        
        # Mở file output nếu được chỉ định
        output_stream = None
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_stream = open(output_file, 'w', encoding='utf-8')
            logger.info(f"Đã mở file output: {output_file}")
        
        # In ra từng đoạn
        for i, chunk in enumerate(chunks):
            # Lấy thông tin về đầu mục
            section_name = chunk.metadata.get("section_name", "Không xác định")
            section_type = chunk.metadata.get("section_type", "Không xác định")
            specialty = chunk.metadata.get("specialty", "")
            
            chunk_info = f"Đầu mục: {section_name}"
            if specialty:
                chunk_info += f" (Chuyên môn: {specialty})"
            
            chunk_content = chunk.page_content
            chunk_size = len(chunk_content)
            
            # In ra console
            print("\n" + "="*80)
            print(f"Chunk #{i+1}/{len(chunks)}: {chunk_info}")
            print(f"Loại: {section_type}, Kích thước: {chunk_size} ký tự")
            print("-"*80)
            print(chunk_content[:1000] + "..." if len(chunk_content) > 1000 else chunk_content)
            print("="*80)
            
            # Ghi vào file nếu được chỉ định
            if output_stream:
                output_stream.write("\n" + "="*80 + "\n")
                output_stream.write(f"Chunk #{i+1}/{len(chunks)}: {chunk_info}\n")
                output_stream.write(f"Loại: {section_type}, Kích thước: {chunk_size} ký tự\n")
                output_stream.write("-"*80 + "\n")
                output_stream.write(chunk_content + "\n")
                output_stream.write("="*80 + "\n")
            
            # Ghi log
            logger.info(f"Chunk #{i+1}: {chunk_size} ký tự, Loại: {section_type}, Đầu mục: {section_name}")
        
        # Đóng file output nếu đã mở
        if output_stream:
            output_stream.close()
            logger.info(f"Đã ghi kết quả chunk vào file: {output_file}")
        
        return chunks, used_split_method, chunk_stats
        
    except Exception as e:
        logger.error(f"Lỗi khi phân chia tài liệu: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def main():
    """Hàm chính để thực hiện chunking tài liệu theo đầu mục."""
    # Tạo parser cho command line arguments
    parser = argparse.ArgumentParser(description="Phân chia tài liệu thành các đoạn theo đầu mục")
    parser.add_argument("--file", "-f", type=str, required=True,
                       help="Đường dẫn đến file cần phân chia")
    parser.add_argument("--output", type=str, default=None,
                       help="Đường dẫn đến file để lưu kết quả phân chia")
    
    # Parse arguments
    args = parser.parse_args()
    
    logger.info("=== BẮT ĐẦU CHUNKING THEO ĐẦU MỤC ===")
    logger.info(f"File: {args.file}")
    
    if args.output:
        logger.info(f"File output: {args.output}")
    
    # Thực hiện chunking
    chunks, split_method, chunk_stats = chunk_document(
        file_path=args.file,
        output_file=args.output
    )
    
    # Hiển thị kết quả
    if chunks:
        print(f"\n{'='*50}")
        print(f"CHUNKING THEO ĐẦU MỤC THÀNH CÔNG")
        print(f"Số lượng đoạn: {chunk_stats['count']}")
        print(f"Phương pháp phân đoạn: {split_method}")
        print(f"Kích thước trung bình: {chunk_stats['avg_size']:.2f} ký tự")
        print(f"Kích thước nhỏ nhất: {chunk_stats['min_size']} ký tự")
        print(f"Kích thước lớn nhất: {chunk_stats['max_size']} ký tự")
        
        print(f"\nPhân bố theo đầu mục:")
        for section_type, count in chunk_stats['sections'].items():
            if count > 0:
                print(f"  {section_type}: {count} chunk")
        
        if chunk_stats['specialties']:
            print(f"\nPhân bố theo chuyên môn:")
            for specialty, count in chunk_stats['specialties'].items():
                print(f"  {specialty}: {count} chunk")
        
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"CHUNKING THẤT BẠI")
        print(f"Kiểm tra file log để biết thêm chi tiết.")
        print(f"{'='*50}")
    
    logger.info("=== KẾT THÚC CHUNKING ===")

if __name__ == "__main__":
    main()