# WSM Project 1: Ranking by Vector Space Models

### Author

114753118 葉子禎

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Basic Command

```bash
python main.py --en_query "planet Taiwan typhoon" --ch_query "音樂 科技"
```

### Command Line Arguments

- `--en_query`: English query string (default: "planet Taiwan typhoon")
- `--ch_query`: Chinese query string (default: "音樂 科技")

## Time Spent on Each Task

| Task   | Description                           | Time Spent |
| ------ | ------------------------------------- | ---------- |
| Task 1 | Vector Space Model (English)          | 1.55s      |
| Task 2 | Relevance Feedback                    | 2.13s      |
| Task 3 | Chinese Text Processing with Jieba    | 1.84s      |
| Task 4 | Evaluation Metrics (MRR, MAP, Recall) | 2.31s      |

## Code Introduction

This project implements a Vector Space Model (VSM) for Information Retrieval with support for both English and Chinese documents.

### Main Components

- **main.py**: Entry point that executes all 4 tasks (VSM ranking, relevance feedback, Chinese processing, evaluation metrics)
- **VectorSpace.py**: Core VSM implementation with Document class, supporting multiple weighting schemes (Raw TF, TF-IDF), similarity metrics (Cosine, Euclidean), and both English/Chinese text processing
- **Parser.py**: Text preprocessing (tokenization, stemming, stop words removal) with language-specific support
- **DocVector.py**: Vector construction using TF and IDF weighting
- **util.py**: Utility functions for cosine similarity and Euclidean distance calculations
- **RankEvaluation.py**: Evaluation metrics implementation (MRR@10, MAP@10, Recall@10)
