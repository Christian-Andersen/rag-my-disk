# rag-my-disk

A CLI tool that indexes your filesystem into a vector database for semantic search.

## Installation

```bash
pip install -e .
```

## Usage

### Index a directory

```bash
python main.py index /path/to/directory --ext .py .md
python main.py index . --recursive
```

Options:
- `--ext`: File extensions to include (can specify multiple)
- `--recursive/--no-recursive`: Search subdirectories (default: true)

### Search the index

```bash
python main.py query your search query
python main.py search your search query -k 5
```

Options:
- `-k`: Number of results to return (default: 3)

### Clear the database

```bash
python main.py clear
```

## Requirements

- Python 3.13+
- See `pyproject.toml` for dependencies