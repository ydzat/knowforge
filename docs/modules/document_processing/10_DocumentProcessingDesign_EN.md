# Document Comprehensive Processing Design

## 1. Background and Purpose

The current KnowForge system requires users to manually categorize different types of inputs (such as PDFs, images, etc.) and cannot automatically identify and process non-textual content (such as images, tables, formulas, etc.) in PDF documents. This limits the system's usability and functional completeness. This document aims to design a document comprehensive processing system that enables KnowForge to automatically identify and process various content types within documents without requiring manual categorization by users.

## 2. Design Goals

1. Automatically identify and extract text, images, tables, and formula content from PDF documents
2. Apply appropriate processing methods for each content type
3. Integrate processing results while maintaining the original document's structure and semantic coherence
4. Seamlessly interface with the existing OCR-LLM-Knowledge base integration
5. Complete implementation before the v1.0 official release

## 3. System Architecture

### 3.1 Core Modules

1. **DocumentAnalyzer**: Document Structure Analyzer
   - Responsible for identifying document structure and content area types
   - Decomposes the document into different types of content blocks

2. **ContentExtractor**: Content Extractor
   - Extracts plain text content
   - Extracts image content
   - Extracts table content
   - Extracts formula content

3. **ContentProcessor**: Content Processor
   - Text processing: directly passed to Splitter
   - Image processing: applies OCR-LLM pipeline
   - Table processing: table recognition and structuring
   - Formula processing: mathematical formula recognition and LaTeX conversion

4. **ContentIntegrator**: Content Integrator
   - Integrates various content types according to their original sequence
   - Maintains the original document structure
   - Generates unified format processing results

### 3.2 Workflow

```
Document Input -> Document Analysis -> Content Extraction -> Type Distribution -> Specialized Processing -> Content Integration -> Unified Output
```

## 4. Detailed Design

### 4.1 DocumentAnalyzer Class

```python
class DocumentAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize document analyzer"""
        
    def analyze_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze document structure, identify content types
        
        Args:
            file_path: Document path
            
        Returns:
            List[Dict]: List of content blocks, each containing type, position, content, etc.
        """
        
    def _identify_content_type(self, content_area) -> str:
        """
        Identify content area type
        
        Returns:
            str: One of 'text', 'image', 'table', 'formula'
        """
```

### 4.2 ContentExtractor Class

```python
class ContentExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize content extractor"""
        
    def extract_text(self, doc, content_block: Dict) -> str:
        """Extract plain text content"""
        
    def extract_image(self, doc, content_block: Dict) -> np.ndarray:
        """Extract image content"""
        
    def extract_table(self, doc, content_block: Dict) -> Dict:
        """Extract table content"""
        
    def extract_formula(self, doc, content_block: Dict) -> Dict:
        """Extract formula content"""
```

### 4.3 ContentProcessor Class

```python
class ContentProcessor:
    def __init__(self, ocr_llm_processor, config: Dict[str, Any] = None):
        """
        Initialize content processor
        
        Args:
            ocr_llm_processor: OCR-LLM processor instance
            config: Configuration options
        """
        
    def process_text(self, text: str) -> str:
        """Process plain text content"""
        
    def process_image(self, image: np.ndarray, context: str = None) -> str:
        """
        Process image content
        
        Args:
            image: Image data
            context: Image context (surrounding text)
        """
        
    def process_table(self, table_data: Dict) -> str:
        """Process table content, returning structured text"""
        
    def process_formula(self, formula_data: Dict) -> str:
        """Process formula content, returning LaTeX or other formats"""
```

### 4.4 ContentIntegrator Class

```python
class ContentIntegrator:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize content integrator"""
        
    def integrate(self, processed_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        Integrate processed content blocks
        
        Args:
            processed_blocks: List of processed content blocks
            
        Returns:
            List[str]: Integrated content paragraphs
        """
        
    def _maintain_structure(self, processed_blocks: List[Dict[str, Any]]) -> List[str]:
        """Maintain original document structure"""
```

## 5. Technology Selection

### 5.1 PDF Processing Technology

- **PyMuPDF (fitz)**: For extracting text, images, and structural information from PDF
- **PDF Element Positioning Algorithm**: Based on page coordinate system to identify text blocks, image blocks, table regions

### 5.2 Table Recognition Technology

- **Camelot**: Python library specifically designed for PDF table extraction
- **Tabula-py**: Another table extraction alternative
- **Custom Table Recognition Algorithm**: Combining boundary box detection and grid line analysis

### 5.3 Formula Recognition Technology

- **Math OCR**: Open-source mathematical formula OCR system
- **MathPix API**: Commercial API, optional for high-quality formula recognition
- **LaTeX Conversion Tools**: Convert recognized formulas to LaTeX format

## 6. Implementation Plan

### 6.1 v0.1.5 (PDF Content Comprehensive Extraction)

- Implement basic DocumentAnalyzer functionality: PDF text and image area recognition
- Implement ContentExtractor's text and image extraction capabilities
- Integrate existing OCR-LLM pipeline to process extracted images

### 6.2 v0.1.6 (Table and Formula Specialized Processing)

- Enhance DocumentAnalyzer: Add table and formula region recognition
- Implement ContentExtractor's table and formula extraction capabilities
- Develop ContentProcessor's table processing functionality
- Implement initial formula recognition and processing capabilities

### 6.3 v0.1.7-v0.2.0 (Content Integration and Format Preservation)

- Perfect all processor functionalities
- Implement ContentIntegrator integration functionality
- Ensure preservation of original document structure in final notes
- Optimize tables and formulas display in different output formats

## 7. Evaluation Metrics

- **Recognition Accuracy**: Content type recognition accuracy > 90%
- **Processing Completeness**: Process at least 95% of document content (no omissions)
- **Structure Preservation**: Output maintains the main structure and information flow of the original document
- **Processing Time**: Average processing time per page < 30 seconds

## 8. Risks and Challenges

1. **Complex Layout Processing**: Accurate recognition of complex layout documents like magazines and academic papers
2. **Non-standard Tables**: Accurate recognition of borderless tables or tables with merged cells
3. **Handwritten Formulas**: Recognition difficulty of handwritten mathematical formulas
4. **Multilingual Processing**: Processing of documents with mixed languages

## 9. Integration with Existing System

This design will integrate with the existing system through:

1. Extending InputHandler to support the new comprehensive processing flow
2. Reusing existing OCR-LLM-Knowledge base integration to process image content
3. Enhancing Splitter to maintain semantic coherence between different content types
4. Modifying OutputWriter to support special display requirements for tables and formulas
