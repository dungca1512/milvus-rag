import os
import logging
import datetime
import argparse
import re
from typing import List, Dict, Any, Optional
import docx2txt
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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

class SkillBasedSplitter:
    """
    Phân chia tài liệu dựa trên các kỹ năng/năng lực quản lý.
    Mỗi chunk sẽ chứa thông tin về một kỹ năng/năng lực cụ thể.
    """
    
    def __init__(self):
        # Các mẫu để nhận diện các kỹ năng cấp cao
        self.skill_patterns = [
            # Mẫu cho năng lực quản lý
            r'^\s*\d+\.\s+(.+)$',                   # 1. Tên kỹ năng
            r'^\s*[A-Z][\s-]+(.+)$',                # A - Tên kỹ năng 
            r'^\s*[A-Z][A-Z\s]+:(.+)$',             # NĂNG LỰC: Tên
        ]
        
        # Mẫu cho cấp độ kỹ năng
        self.skill_level_pattern = r'^\s*Cấp độ\s+(\d+)(.*)$'
    
    def extract_skills(self, text):
        """
        Trích xuất các kỹ năng/năng lực từ văn bản và vị trí của chúng.
        
        Returns:
            List of tuples (start_pos, end_pos, skill_name, skill_content)
        """
        lines = text.split('\n')
        skills = []
        current_line_pos = 0
        skill_start = -1
        current_skill = ""
        
        for i, line in enumerate(lines):
            line_start_pos = current_line_pos
            current_line_pos += len(line) + 1  # +1 for newline character
            
            # Kiểm tra xem dòng hiện tại có phải là tên kỹ năng mới không
            is_skill_header = False
            skill_name = ""
            
            for pattern in self.skill_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    is_skill_header = True
                    skill_name = match.group(1).strip() if match.groups() else line.strip()
                    break
            
            # Nếu tìm thấy kỹ năng mới
            if is_skill_header:
                # Nếu đã có kỹ năng trước đó, lưu lại
                if skill_start >= 0 and current_skill:
                    skill_content = "\n".join(lines[skill_start:i])
                    skills.append((skill_start, line_start_pos, current_skill, skill_content))
                
                # Bắt đầu kỹ năng mới
                skill_start = i
                current_skill = skill_name
        
        # Thêm kỹ năng cuối cùng nếu có
        if skill_start >= 0 and current_skill:
            skill_content = "\n".join(lines[skill_start:])
            skills.append((skill_start, current_line_pos, current_skill, skill_content))
        
        return skills
    
    def extract_skill_levels(self, skill_content):
        """
        Tách nội dung kỹ năng thành các cấp độ khác nhau.
        
        Returns:
            List of tuples (level, level_content)
        """
        lines = skill_content.split('\n')
        levels = []
        current_level = None
        current_level_content = []
        
        for line in lines:
            # Kiểm tra xem dòng hiện tại có phải là định nghĩa cấp độ mới không
            level_match = re.match(self.skill_level_pattern, line.strip())
            
            if level_match:
                # Nếu đã có cấp độ trước đó, lưu lại
                if current_level is not None and current_level_content:
                    level_text = "\n".join(current_level_content)
                    levels.append((current_level, level_text))
                
                # Bắt đầu cấp độ mới
                current_level = int(level_match.group(1))
                current_level_content = [line]
            elif current_level is not None:
                # Thêm dòng vào cấp độ hiện tại
                current_level_content.append(line)
        
        # Thêm cấp độ cuối cùng nếu có
        if current_level is not None and current_level_content:
            level_text = "\n".join(current_level_content)
            levels.append((current_level, level_text))
        
        return levels
    
    def split_documents(self, documents):
        """
        Phân chia documents thành các chunk theo từng kỹ năng/năng lực.
        """
        result = []
        
        for doc in documents:
            text = doc.page_content
            base_metadata = doc.metadata
            
            # Trích xuất các kỹ năng từ tài liệu
            skills = self.extract_skills(text)
            
            if not skills:
                # Nếu không tìm thấy kỹ năng nào, giữ nguyên document
                result.append(doc)
                continue
            
            # Tạo document cho từng kỹ năng
            for _, _, skill_name, skill_content in skills:
                # Tạo metadata cho kỹ năng
                skill_metadata = base_metadata.copy()
                skill_metadata["skill_name"] = skill_name
                
                # Kiểm tra xem có các cấp độ kỹ năng không
                skill_levels = self.extract_skill_levels(skill_content)
                
                if skill_levels:
                    # Nếu có các cấp độ, tạo document cho từng cấp độ
                    for level, level_content in skill_levels:
                        level_metadata = skill_metadata.copy()
                        level_metadata["skill_level"] = level
                        
                        level_doc = Document(
                            page_content=level_content,
                            metadata=level_metadata
                        )
                        result.append(level_doc)
                else:
                    # Nếu không có cấp độ, tạo document cho toàn bộ kỹ năng
                    skill_doc = Document(
                        page_content=skill_content,
                        metadata=skill_metadata
                    )
                    result.append(skill_doc)
        
        return result

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Phân chia danh sách document theo từng năng lực quản lý.
    
    Args:
        documents (List[Document]): Danh sách document cần phân chia
        chunk_size: Không được sử dụng, chỉ giữ để tương thích với interface
        chunk_overlap: Không được sử dụng, chỉ giữ để tương thích với interface
        
    Returns:
        List[Document]: Danh sách document đã được phân chia
        str: Phương pháp phân chia đã sử dụng
    """
    if not documents:
        return [], "none"
    
    # Sử dụng SkillBasedSplitter để phân chia theo kỹ năng
    splitter = SkillBasedSplitter()
    method = "skill_based"
    
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
            "skills": {}
        }
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    
    # Đếm số lượng chunk theo kỹ năng
    skill_counts = {}
    for chunk in chunks:
        skill_name = chunk.metadata.get("skill_name", "Unknown")
        if skill_name in skill_counts:
            skill_counts[skill_name] += 1
        else:
            skill_counts[skill_name] = 1
    
    return {
        "count": len(chunks),
        "avg_size": avg_size,
        "min_size": min_size,
        "max_size": max_size,
        "skills": skill_counts
    }

def chunk_document(file_path, output_file=None):
    """
    Đọc và phân chia tài liệu thành các đoạn theo từng năng lực quản lý.
    In ra từng đoạn chunk được tạo ra.
    
    Args:
        file_path (str): Đường dẫn đến tài liệu cần xử lý
        output_file (str, optional): Đường dẫn đến file để lưu kết quả chunk
        
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
            # Lấy thông tin về kỹ năng
            skill_name = chunk.metadata.get("skill_name", "Không xác định")
            skill_level = chunk.metadata.get("skill_level", "N/A")
            
            chunk_info = f"Kỹ năng: {skill_name}"
            if skill_level != "N/A":
                chunk_info += f", Cấp độ: {skill_level}"
            
            chunk_content = chunk.page_content
            chunk_size = len(chunk_content)
            
            # In ra console
            print("\n" + "="*80)
            print(f"Chunk #{i+1}/{len(chunks)}: {chunk_info} (Kích thước: {chunk_size} ký tự)")
            print("-"*80)
            print(chunk_content)
            print("="*80)
            
            # Ghi vào file nếu được chỉ định
            if output_stream:
                output_stream.write("\n" + "="*80 + "\n")
                output_stream.write(f"Chunk #{i+1}/{len(chunks)}: {chunk_info} (Kích thước: {chunk_size} ký tự)\n")
                output_stream.write("-"*80 + "\n")
                output_stream.write(chunk_content + "\n")
                output_stream.write("="*80 + "\n")
            
            # Ghi log
            logger.info(f"Chunk #{i+1}: {chunk_size} ký tự, Kỹ năng: {skill_name}, Cấp độ: {skill_level}")
        
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
    """Hàm chính để thực hiện chunking tài liệu theo năng lực quản lý."""
    # Tạo parser cho command line arguments
    parser = argparse.ArgumentParser(description="Phân chia tài liệu thành các đoạn theo năng lực quản lý")
    parser.add_argument("--file", "-f", type=str, required=True,
                       help="Đường dẫn đến file cần phân chia")
    parser.add_argument("--output", type=str, default=None,
                       help="Đường dẫn đến file để lưu kết quả phân chia")
    
    # Parse arguments
    args = parser.parse_args()
    
    logger.info("=== BẮT ĐẦU CHUNKING THEO NĂNG LỰC QUẢN LÝ ===")
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
        print(f"CHUNKING THEO NĂNG LỰC THÀNH CÔNG")
        print(f"Số lượng đoạn: {chunk_stats['count']}")
        print(f"Phương pháp phân đoạn: {split_method}")
        print(f"Kích thước trung bình: {chunk_stats['avg_size']:.2f} ký tự")
        print(f"Kích thước nhỏ nhất: {chunk_stats['min_size']} ký tự")
        print(f"Kích thước lớn nhất: {chunk_stats['max_size']} ký tự")
        
        print(f"\nCác năng lực đã phân chia:")
        for skill_name, count in chunk_stats['skills'].items():
            print(f"  {skill_name}: {count} chunk")
        
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"CHUNKING THẤT BẠI")
        print(f"Kiểm tra file log để biết thêm chi tiết.")
        print(f"{'='*50}")
    
    logger.info("=== KẾT THÚC CHUNKING ===")

if __name__ == "__main__":
    main()