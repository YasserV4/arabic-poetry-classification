# Arabic Poetry Era Classification

## 📖 Solution Description

This project implements a **BERT-based text classification model** to classify Arabic poetry based on the historical era during which the poet lived and wrote. The solution leverages the power of transformer models to understand the contextual nuances in classical Arabic poetry and accurately predict the historical period of each poem.

### Key Features:
- **Deep Learning Approach**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for contextual understanding of Arabic text
- **Multi-class Classification**: Classifies poetry into 9 distinct historical eras
- **Comprehensive Analysis**: Includes EDA, data preprocessing, model training, evaluation, and error analysis
- **Performance Optimization**: Implements class weighting to handle imbalanced dataset

### Model Performance:
- **Total Accuracy**: [28]%
- **Weighted F1-Score**: [30]
- **Training Dataset**: 74,988 poetry samples across 9 historical eras

---

## 📊 Data Description

- **link**" https://www.kaggle.com/datasets/mdanok/arabic-poetry-dataset/data

### Dataset Overview
- **Source**: Arabic Poetry Dataset
- **Total Records**: 74,988 poetry samples
- **Features**: 
  - `poet_name`: Name of the poet (755 unique poets)
  - `poet_era`: Historical era label (9 categories)
  - `poem_tags`: Poetry tags/themes (6,033 unique tags)
  - `poem_title`: Title of the poem (73,502 unique titles)
  - `poem_text`: Full text of the poetry (74,581 unique texts)
  - `poem_count`: 90% of it ranges between 1-13 (74,581 unique poem counts)

### Historical Eras Distribution
The dataset covers 9 distinct historical periods of Arabic poetry:

| Era (Arabic) | Era (English) | Sample Count | Percentage |
|--------------|---------------|--------------|------------|
| العصر العباسي | Abbasid Era | 26,723 | 35.64% |
| العصر المملوكي | Mamluk Era | 13,085 | 17.45% |
| العصر الايوبي | Ayyubid Era | 8,157 | 10.88% |
| العصر العثماني | Ottoman Era | 7,545 | 10.06% |
| العصر الاموي | Umayyad Era | 7,330 | 9.77% |
| العصر الأندلسي | Andalusian Era | 6,171 | 8.23% |
| المخضرمون | Mukhadram Era | 3,290 | 4.39% |
| العصر الجاهلي | Pre-Islamic Era | 2,350 | 3.13% |
| العصر الاسلامي | Islamic Era | 337 | 0.45% |

### Data Characteristics
- **Class Imbalance**: Significant imbalance with Abbasid Era representing 35.64% of data while Islamic Era only 0.45%
- **Text Complexity**: Classical Arabic poetry with varying lengths and styles

---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU atleast 8GB of vRAM
- At least 8GB RAM

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YasserV4/arabic-poetry-classification.git
   cd arabic-poetry-classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
The project uses the following main libraries:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library for BERT
- `scikit-learn` - Machine learning utilities and evaluation metrics
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CPU**: Multi-core processor for data preprocessing
- **Memory**: Minimum 8GB RAM, 16GB+ recommended

---

## 📋 Project Structure

```
arabic-poetry-classification/
├── README.md
├── requirements.txt
├── arabic_poetry_bert_model/
│   ├── config.json
│   ├── label_encoder.pkl
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── dataset/
│   └── Arabic_Poetry_Dataset.csv
├── results/
│   ├── checkpoint-6500/
│   └── checkpoint-29700/
└── EDA.ipynb
```

---

## How to Run the Project

### Option 1: Jupyter Notebook (Recommended)
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `EDA.ipynb`
3. Run all cells sequentially

### Option 2: Run Individual Components
The project contains trained model checkpoints and can be used for inference:
- **Model files**: Located in `arabic_poetry_bert_model/` directory
- **Training checkpoints**: Available in `results/checkpoint-6500/` and `results/checkpoint-29700/`
- **Dataset**: `dataset/Arabic_Poetry_Dataset.csv`

### Configuration
- Modify hyperparameters in the configuration section of the notebook
- Adjust batch size based on your GPU memory
- Set appropriate learning rate and epochs for your hardware

---

## 🎯 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Distribution Analysis**: Visualization of era distribution and class imbalance
- **Text Length Analysis**: Distribution of poem lengths across different eras
- **Poet Analysis**: Most prolific poets per era

### 2. Data Preprocessing
- **Text Cleaning**: Removed the HARAKAT from the poems, encoded the era, separate poems tags
- **Tokenization**: BERT tokenizer for Arabic text
- **Label Encoding**: Numerical encoding of historical eras
- **Data Splitting**: Train/Validation/Test split with stratification

### 3. Model Selection and Justification
**Chosen Model**: BERT (Bidirectional Encoder Representations from Transformers)

**Justification**:
- **Contextual Understanding**: BERT's bidirectional nature captures context from both directions
- **Arabic Language Support**: Pre-trained Arabic BERT models available
- **Sequence Classification**: Well-suited for text classification tasks
- **Transfer Learning**: Leverages pre-trained knowledge for better performance
- **State-of-the-art**: Proven effectiveness in NLP tasks

### 4. Training Strategy
- **Class Weighting**: Applied to handle class imbalance
- **Fine-tuning**: Fine-tuned pre-trained BERT model

---

## 📈 Results and Performance

### Model Performance Metrics
- **Overall Accuracy**: [28]%
- **Weighted F1-Score**: [30]

---

## 💡 Recommendations for Improvement

### Model Improvements
1. **Advanced Class Balancing Techniques**:
   - **SMOTE (Synthetic Minority Oversampling Technique)**: Generate synthetic samples for minority classes
   - **Focal Loss**: Implement focal loss function to focus learning on hard-to-classify samples
   - **Class-balanced Loss**: Use inversely weighted loss functions based on class frequency
   - **Ensemble with Balanced Sampling**: Train multiple models with different balanced subsets

2. **Data Augmentation Strategies**:
   - **Back-translation**: Translate Arabic text to another language and back to create variations
   - **Synonym Replacement**: Replace words with synonyms while preserving meaning
   - **Text Paraphrasing**: Generate paraphrased versions of minority class samples
   - **Mixup/CutMix**: Apply data mixing techniques for text augmentation

3. **Advanced Model Architectures**:
   - **Ensemble Methods**: Combine multiple BERT models with different initialization seeds
   - **Hierarchical Classification**: First classify by broader time periods, then fine-tune within periods
   - **Multi-task Learning**: Train on related tasks simultaneously (poet identification, theme classification)
   - **Domain-Adaptive Pre-training**: Further pre-train BERT on classical Arabic poetry corpus

4. **Sampling Strategies**:
   - **Stratified Sampling**: Ensure balanced representation in train/validation splits
   - **Cost-sensitive Learning**: Assign higher misclassification costs to minority classes
   - **Threshold Optimization**: Optimize classification thresholds per class for better balance

### Data Improvements
1. **Targeted Data Collection**:
   - **Minority Class Expansion**: Actively collect more samples from underrepresented eras (العصر الاسلامي, العصر الجاهلي, المخضرمون)
   - **Quality Verification**: Implement expert validation for historical era attribution accuracy
   - **Multi-source Integration**: Combine data from multiple Arabic poetry repositories

2. **Feature Engineering**:
   - **Temporal Features**: Extract decade/century information as additional features
   - **Linguistic Features**: Add classical Arabic linguistic markers (meter, rhyme patterns)
   - **Poet-specific Features**: Include poet biographical information and writing style characteristics
   - **Historical Context Features**: Incorporate historical events and cultural markers

3. **Preprocessing Enhancements**:
   - **Advanced Text Normalization**: Standardize classical Arabic variations and diacritics
   - **Noise Reduction**: Remove or standardize modern Arabic interpolations
   - **Segmentation Optimization**: Experiment with different text chunking strategies for long poems
   - **Cross-validation Strategy**: Implement poet-aware splits to prevent data leakage


---

## 🔮 Assumptions Made

1. **Text Quality**: Assumption that the poetry texts are accurately transcribed and attributed
2. **Era Classification**: Poets are correctly classified into their respective historical eras
3. **Language Consistency**: Classical Arabic features remain consistent within eras
4. **Data Completeness**: Missing values are randomly distributed and not systematic

---

## 📁 File Descriptions

- **`EDA.ipynb`**: Main Jupyter notebook containing the complete solution with exploratory data analysis
- **`requirements.txt`**: List of required Python packages
- **`dataset/Arabic_Poetry_Dataset.csv`**: The Arabic poetry dataset
- **`arabic_poetry_bert_model/`**: Directory containing the trained BERT model files
- **`results/`**: Directory containing training checkpoints