# Enhanced Document Categorization System

This guide explains how to use the new enhanced categorization system that implements chunking, TF-IDF weighted embeddings, and intelligent clustering for much better document organization.

## 🚀 What's New

### Key Improvements Over Original System

1. **Chunking**: Large documents are split into semantic chunks (2000 chars with 200 overlap) for better embedding quality
2. **TF-IDF Weighting**: Chunk importance is computed using TF-IDF, so important parts of documents count more in the final embedding
3. **Advanced Clustering**: Support for both auto-clustering (HDBSCAN+UMAP) and fixed-K clustering (KMeans + silhouette optimization)
4. **Meaningful Labels**: Cluster names generated from TF-IDF top terms for human-readable categories
5. **Incremental Classification**: Save centroids to classify new files without re-clustering everything
6. **Better Content Understanding**: Enhanced semantic analysis with multiple signals

## 📦 Installation

### Required Dependencies
```bash
pip install -r requirements_enhanced_categorization.txt
```

### Optional Advanced Dependencies (for better clustering)
```bash
pip install umap-learn hdbscan keybert yake bertopic
```

## 🔧 Usage

### 1. Basic Batch Categorization

Categorize a folder of documents:

```bash
# Auto-determine number of categories
python smart_categorize_v2.py --source ./my_documents --dest ./categorized_output

# Force specific number of categories
python smart_categorize_v2.py --source ./my_documents --dest ./categorized_output --k 5

# Use advanced clustering (requires umap-learn and hdbscan)
python smart_categorize_v2.py --source ./my_documents --dest ./categorized_output --use_hdbscan
```

### 2. Incremental Classification

Classify new files using a previously trained model:

```bash
# Classify new files into existing categories
python incremental_classifier.py --new_files ./new_docs --model_dir ./categorized_output --output ./classified

# Adjust similarity threshold (0.6-0.8 recommended)
python incremental_classifier.py --new_files ./new_docs --model_dir ./categorized_output --output ./classified --threshold 0.7
```

### 3. Programmatic Usage

```python
from smart_categorize_v2 import EnhancedCategorizer

# Initialize categorizer
categorizer = EnhancedCategorizer()

# Categorize single content
category = categorizer.categorize_content("Your document content here")
print(f"Category: {category}")

# Load saved model for incremental classification
categorizer.load_saved_model("./categorized_output")
new_category = categorizer.categorize_content("New document content")
```

### 4. Enhanced Embed Service

Start the enhanced embedding service with categorization:

```bash
cd services/embed
python enhanced_app.py
```

API endpoints:
- `POST /categorize_content` - Categorize text content
- `POST /process_file` - Process a single file (extract, chunk, embed, categorize)
- `POST /batch_categorize` - Batch categorize multiple files
- `POST /load_categorization_model` - Load a saved model
- `GET /model_info` - Get current model information

## 📊 Understanding the Output

### Folder Structure
After categorization, you'll get:
```
categorized_output/
├── cat0_research_paper_analysis/     # Category 0: Research papers
├── cat1_invoice_payment_bill/        # Category 1: Financial documents  
├── cat2_meeting_agenda_minutes/      # Category 2: Meeting documents
├── noise_uncategorized/              # Uncategorized/noise documents
├── centroids.npy                     # Saved centroids for incremental classification
├── file_embeddings.npy               # All file embeddings
├── classification_metadata.json      # Model metadata
├── categories_map.json               # File -> category mapping
└── tfidf_vectorizer.pkl             # TF-IDF vectorizer for chunk weighting
```

### Category Names
Categories are automatically named using the top TF-IDF terms:
- `cat0_research_paper_analysis` = Category 0 with top terms: research, paper, analysis
- `cat1_invoice_payment_bill` = Category 1 with top terms: invoice, payment, bill

### Incremental Classification Results
```
classified/
├── cat0_research_paper_analysis/     # Files classified into existing categories
├── cat1_invoice_payment_bill/
├── review/                           # Files needing manual review
│   ├── no_text/                     # Files with no extractable text
│   └── error/                       # Files with processing errors
└── classification_log.json          # Detailed classification log
```

## 🎯 Integration with Clarifile

### 1. Update Parser Service

Replace the categorization in your parser service:

```python
# In services/parser/app.py
from smart_categorize_v2 import EnhancedCategorizer

categorizer = EnhancedCategorizer()
# Load your trained model
categorizer.load_saved_model("path/to/your/model")

# Use in your parsing pipeline
category = categorizer.categorize_content(extracted_text)
```

### 2. Google Drive Integration

For your browser extension, you can:

1. **Initial Setup**: Run batch categorization on existing Drive files
2. **Real-time**: Use incremental classification for new files
3. **API Integration**: Use the enhanced embed service endpoints

### 3. Database Integration

Update your database schema to store:
- Chunk embeddings (for better search)
- Category confidence scores
- TF-IDF importance weights

## 🔧 Configuration & Tuning

### Chunk Size Optimization
- **Small documents** (< 1000 chars): Use chunk_size=500, overlap=50
- **Medium documents** (1000-5000 chars): Use chunk_size=1000, overlap=100  
- **Large documents** (> 5000 chars): Use chunk_size=2000, overlap=200

### Similarity Thresholds
- **Conservative** (fewer false positives): threshold=0.7-0.8
- **Balanced**: threshold=0.65-0.7
- **Aggressive** (fewer manual reviews): threshold=0.6-0.65

### Clustering Parameters
- **Auto-clustering**: Use `--use_hdbscan` for variable number of categories
- **Fixed categories**: Use `--k N` when you know how many categories you want
- **Silhouette optimization**: Let the system choose optimal K (default)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_enhanced_categorization_v2.py
```

This will test:
- Text extraction and chunking
- Enhanced categorization
- Batch processing
- Incremental classification

## 📈 Performance Tips

### 1. Batch Processing
- Process files in batches of 100-1000 for optimal memory usage
- Use GPU if available (install `sentence-transformers[gpu]`)

### 2. Caching
- Cache embeddings for frequently accessed documents
- Store TF-IDF vectorizer for consistent chunk weighting

### 3. Monitoring
- Track classification confidence scores
- Monitor files sent to "review" folder
- Use silhouette scores to evaluate clustering quality

## 🔍 Troubleshooting

### Common Issues

1. **"No text extracted"**
   - Enable OCR: Install `pdf2image` and `pytesseract`
   - Check file formats are supported
   - Verify file permissions

2. **Poor categorization quality**
   - Increase chunk size for longer documents
   - Adjust similarity threshold
   - Add more training documents to underrepresented categories

3. **Memory issues**
   - Reduce batch size
   - Use smaller embedding models
   - Process files in chunks

4. **Slow performance**
   - Install FAISS for faster similarity search
   - Use GPU acceleration
   - Cache embeddings

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Next Steps

1. **Train on your data**: Run batch categorization on your Google Drive files
2. **Evaluate results**: Check category coherence and adjust parameters
3. **Deploy incremental**: Set up real-time classification for new files
4. **Monitor performance**: Track classification accuracy and user feedback
5. **Iterate**: Retrain periodically with new documents and user corrections

## 📚 Advanced Features

### Custom Category Prototypes
You can define custom category prototypes for domain-specific classification:

```python
categorizer.category_prototypes = {
    "Legal: Contract": "contract agreement legal terms conditions law",
    "Medical: Report": "patient diagnosis treatment medical clinical",
    "Technical: API": "api documentation endpoint request response"
}
```

### Human-in-the-Loop Training
1. Review files in the "review" folder
2. Manually categorize them
3. Use these as training data for supervised classification
4. Train a logistic regression model on embeddings for faster classification

### Multi-language Support
Use multilingual models:
```python
categorizer = EnhancedCategorizer("paraphrase-multilingual-MiniLM-L12-v2")
```

This enhanced system provides a solid foundation for intelligent document organization that actually understands content, not just filenames or metadata.
