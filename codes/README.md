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
