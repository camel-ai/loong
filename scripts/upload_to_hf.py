import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import backoff
from huggingface_hub.utils import HfHubHTTPError
import requests

# CAMEL imports
from camel.datasets import DataPoint, StaticDataset
from camel.datahubs.huggingface import HuggingFaceDatasetManager
from camel.datahubs.models import Record

# Import Hugging Face Dataset
from datasets import Dataset as HFDataset


def split_train_test(data, train_ratio, random_seed=42):
    """
    Split dataset into training and testing sets.

    Args:
        data (list): List of data entries to split
        train_ratio (float): Ratio of training data
        random_seed (int): Random seed for reproducibility, defaults to 42

    Returns:
        tuple: (training_data, testing_data)
    """
    import random
    random.seed(random_seed)
    
    # Copy data to avoid modifying original
    data_copy = data.copy()
    # Shuffle data randomly
    random.shuffle(data_copy)
    
    # Calculate training set size
    train_size = int(len(data_copy) * train_ratio)
    
    # Split data
    train_data = data_copy[:train_size]
    test_data = data_copy[train_size:]
    
    return train_data, test_data


def load_dataset_files(file_paths):
    """
    Load and parse multiple JSON dataset files.

    Args:
        file_paths (list): List of paths to the JSON dataset files.

    Returns:
        list: Combined list of data from all files.
    """
    combined_data = []
    
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            combined_data.extend(data)
        else:
            # If the data is not a list, it might be a dictionary with a key containing the list
            for key, value in data.items():
                if isinstance(value, list):
                    combined_data.extend(value)
                    break
    
    print(f"Loaded {len(combined_data)} total entries")
    return combined_data

def transform_data_to_datapoints(data_entries: List[Dict[str, Any]]) -> List[DataPoint]:
    """
    Transform loong data into DataPoint objects for local use.

    Args:
        data_entries (List[Dict[str, Any]]): List of data dictionaries.

    Returns:
        List[DataPoint]: List of DataPoint objects.
    """
    datapoints = []
    for entry in data_entries:
        # Extract fields
        question = entry.get('question', '')
        rationale = entry.get('rationale', '')
        final_answer = entry.get('final_answer', '')
        metadata = entry.get('metadata', {})
        # stringify metadata from a dict to a string
        meta_data_str = json.dumps(metadata)
        # Create DataPoint
        datapoint = dict(
            question=question,
            rationale=rationale,
            final_answer=final_answer,
            meta_data=meta_data_str,
        )
        datapoints.append(datapoint)
    
    return datapoints

def convert_datapoints_to_hf_dataset(datapoints: List[DataPoint]) -> HFDataset:
    """Convert a list of DataPoint objects to a Hugging Face Dataset."""
    # Convert DataPoints to dictionaries
    data_dicts = [dp for dp in datapoints]

    # Create a Hugging Face Dataset
    hf_dataset = HFDataset.from_list(data_dicts)
    return hf_dataset

def transform_data_to_records(data_entries: List[Dict[str, Any]]) -> List[Record]:
    """
    Transform loong data into Record objects for Hugging Face upload.

    Args:
        data_entries (List[Dict[str, Any]]): List of data dictionaries.

    Returns:
        List[Record]: List of Record objects.
    """
    records = []
    
    for id, entry in enumerate(data_entries):
        # Extract fields
        question = entry.get('question', '')
        final_answer = entry.get('final_answer', '')
        rationale = entry.get('rationale', '')
        meta_data = entry.get('metadata', '')

        # read back from metadata string
        domain = meta_data.get('domain', '')

        record = Record(
            source_type=domain,
            question=question,
            final_answer=final_answer,
            rationale=rationale,
            meta_data=json.dumps(meta_data),
        )
        records.append(record)
    
    return records

def generate_or_validate_dataset_name(username, dataset_name=None):
    """
    Generates a default dataset name or validates and formats a user-provided name.

    Args:
        username (str): Hugging Face username.
        dataset_name (str, optional): User-provided custom dataset name.

    Returns:
        str: Formatted dataset name.
    """
    if not dataset_name:
        dataset_name = "loong"
    # Format the dataset name to include the username
    return f"{username}/{dataset_name}"

def create_dataset(manager, dataset_name):
    """
    Creates a new dataset on Hugging Face and returns the dataset URL.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.

    Returns:
        str: URL of the created dataset.
    """
    dataset_url = manager.create_dataset(dataset_name)
    return dataset_url

def create_dataset_card(manager, dataset_name, username):
    """
    Creates a dataset card to add metadata.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.
        username (str): Hugging Face username.
    """
    manager.create_dataset_card(
        dataset_name=dataset_name,
        description="A comprehensive collection of 8,729 high-quality problems across 12 diverse domains, curated for Project Loong. Each problem includes a detailed executable rationale and solution, designed for training and evaluating reasoning models.",
        license="mit",  # Using lowercase 'mit' as required by HuggingFace
        tags=["reasoning", "problem-solving", "project-loong", "multi-domain", "mathematics", "physics", "chemistry", "finance", "optimization"],
        authors=[username],
        language=["en"],
        task_categories=["question-answering"],
        content="# Project Loong Seed Dataset\n\n"
                "This dataset is part of Project Loong, a collaborative effort to explore whether reasoning-capable models can bootstrap themselves from small, high-quality seed datasets by generating synthetic data and verifying LLM agent responses.\n\n"
                "## Dataset Description\n\n"
                "This comprehensive collection contains 8,729 human-vetted problems across 12 diverse domains:\n\n"
                "- 🧮 **Advanced Math:** 1,611 questions\n"
                "- ⚛️ **Advanced Physics:** 429 questions\n"
                "- 🧪 **Chemistry:** 3,076 questions\n"
                "- 🧬 **Computational Biology:** 51 questions\n"
                "- 💹 **Finance:** 235 questions\n"
                "- 🎮 **Game:** 926 questions\n"
                "- 📈 **Graph & Discrete Math:** 178 questions\n"
                "- 🧠 **Logic:** 130 questions\n"
                "- 📐 **Mathematical Programming:** 76 questions\n"
                "- 💊 **Medicine:** 916 questions\n"
                "- 🔒 **Security & Safety:** 516 questions\n"
                "- 🧑‍💻 **Programming:** 585 questions\n\n"
                "## Data Structure\n\n"
                "Each entry includes:\n\n"
                "- A problem statement\n"
                "- A detailed rationale explaining the solution approach\n"
                "- The final answer or solution\n"
                "- Metadata including problem ID, domain information, and other relevant attributes\n\n"
                "## Dataset Purpose\n\n"
                "Each dataset is designed to allow automatic evaluation via verifiers, usually by executing the rationale code and comparing the output to the known answer. This collection serves as seed data for exploring whether reasoning-capable models can bootstrap themselves by generating and verifying synthetic data."
    )

def add_records_to_dataset(manager, dataset_name, records):
    """
    Adds a list of Record objects to the dataset.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.
        records (list): List of Record objects.
    """
    manager.add_records(dataset_name, records)

# Add retry decorator for API calls
@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.HTTPError),
    max_tries=5,
    max_time=300,
    giveup=lambda e: e.response.status_code not in [429, 500, 502, 503, 504] if hasattr(e, 'response') else False
)
def push_to_hub_with_retry(dataset, dataset_name, config_name, split):
    """
    Push dataset to hub with retry mechanism.
    """
    dataset.push_to_hub(
        dataset_name,
        config_name=config_name,
        split=split,
        private=False,
        token=os.environ.get("HF_TOKEN")
    )
    # Add delay after successful push
    time.sleep(5)

def upload_domain_dataset(data_entries: List[Dict[str, Any]], 
                        username: str, 
                        base_dataset_name: str,
                        domain: str,
                        split: str):
    """
    Upload dataset for a specific domain and split.

    Args:
        data_entries (List[Dict[str, Any]]): Data entries for the domain.
        username (str): Hugging Face username.
        base_dataset_name (str): Base name for the dataset.
        domain (str): Domain name.
        split (str): Split name ('train' or 'test').
    """
    try:
        # Generate dataset name in format: username/base_dataset_name
        dataset_name = f"{username}/{base_dataset_name}"
        
        # Transform data while keeping original structure
        formatted_data = []
        for entry in data_entries:
            # Ensure metadata is in string format
            metadata = entry.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            # Add domain to metadata
            metadata['domain'] = domain
            
            # Convert metadata to string
            metadata_str = json.dumps(metadata)
            
            formatted_data.append({
                'question': str(entry.get('question', '')),
                'rationale': str(entry.get('rationale', '')),
                'final_answer': str(entry.get('final_answer', '')),
                'metadata': metadata_str,
                'domain': domain
            })
        
        # Create HuggingFace Dataset
        hf_dataset = HFDataset.from_list(formatted_data)
        
        # Push to hub with retry mechanism
        push_to_hub_with_retry(hf_dataset, dataset_name, domain, split)
        
        print(f"Successfully uploaded {domain} {split} split")
        
    except Exception as e:
        print(f"Error uploading {domain} {split} split: {e}")
        raise

def create_domain_dataset_card(manager, dataset_name, username):
    """
    Creates a dataset card for the entire dataset.

    Args:
        manager (HuggingFaceDatasetManager): Dataset manager instance.
        dataset_name (str): Name of the dataset.
        username (str): Hugging Face username.
    """
    domain_descriptions = {
        "advanced_math": "Advanced mathematics problems including calculus, algebra, and number theory",
        "advanced_physics": "Physics problems covering mechanics, thermodynamics, and quantum physics",
        "chemistry": "Chemistry problems covering organic, inorganic, physical chemistry and chemical reactions",
        "computational_biology": "Biological computation and analysis problems",
        "finance": "Financial analysis and modeling problems",
        "graph_discrete_math": "Graph theory and discrete mathematics problems",
        "logic": "Logical reasoning and proof problems",
        "mathematical_programming": "Optimization and mathematical programming problems",
        "security_and_safety": "Security and safety analysis problems",
        "medicine": "Medicine and biology problems",
        "programming": "Programming problems"
    }

    content = """# Project Loong Dataset

This dataset is part of Project Loong, a collaborative effort to explore whether reasoning-capable models can bootstrap themselves from small, high-quality seed datasets.

## Dataset Description

This comprehensive collection contains problems across multiple domains, each split is determined by the domain.

### Available Domains:

"""
    
    # Add domain descriptions
    for domain, desc in domain_descriptions.items():
        content += f"### {domain.replace('_', ' ').title()}\n{desc}\n\n"
    
    content += """
## Data Structure

Each entry includes:
- A problem statement
- A detailed rationale explaining the solution approach
- The final answer or solution
- Metadata including problem ID and domain information
- Domain label

## Usage

```python
from datasets import load_dataset

# Load a specific domain's data
domain = "advanced_math"  # or any other domain
dataset = load_dataset("camel-ai/loong", domain)

# Access specific splits
train_data = dataset["train"]
test_data = dataset["test"]
```
"""
    
    # Create dataset card
    manager.create_dataset_card(
        dataset_name=dataset_name,
        description="A comprehensive collection of high-quality problems across diverse domains, curated for Project Loong. Each problem includes a detailed executable rationale and solution.",
        license="mit",
        tags=["reasoning", "problem-solving", "project-loong", "multi-domain", "mathematics", "physics", "chemistry", "finance", "optimization"],
        authors=[username],
        language=["en"],
        task_categories=["question-answering"],
        content=content
    )

def main():
    # Get the parent directory of the current script (i.e., loong directory)
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "data"
    
    print(f"Looking for data in: {data_dir}")
    
    # Define domain paths and their corresponding train/test split ratios
    # Higher ratio means more training data, lower ratio means more test data
    # For example:
    # - 0.7: 70% training, 30% testing (standard split)
    # - 0.8: 80% training, 20% testing (for domains with less data)
    # - 0.6: 60% training, 40% testing (for domains needing more testing)
    domain_configs = {
        "advanced_math": {
            "path": data_dir / "advanced_math" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "advanced_physics": {
            "path": data_dir / "advanced_physics" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "chemistry": {
            "path": data_dir / "chemistry" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "computational_biology": {
            "path": data_dir / "computational_biology" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "finance": {
            "path": data_dir / "finance" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "games": {
            "path": data_dir / "games" / "blackjack" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "graph_discrete_math": {
            "path": data_dir / "graph_discrete_math" / "seed_dataset.json",
            "train_ratio": 0.5  # Standard split
        },
        "logic": {
            "path": data_dir / "logic" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "mathematical_programming": {
            "path": data_dir / "mathematical_programming" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "security_and_safety": {
            "path": data_dir / "security_and_safety" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "medicine": {
            "path": data_dir / "medicine" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        },
        "programming": {
            "path": data_dir / "programming" / "seed_dataset.json",
            "train_ratio": 0.3  # Standard split
        }
    }

    # When using the configurations:
    domain_paths = {domain: config["path"] for domain, config in domain_configs.items()}

    username = "camel-ai"
    base_dataset_name = "loong"
    
    # First create the main dataset (if it doesn't exist)
    manager = HuggingFaceDatasetManager()
    dataset_name = f"{username}/{base_dataset_name}"
    try:
        create_dataset(manager, dataset_name)
        create_domain_dataset_card(manager, dataset_name, username)
    except Exception as e:
        print(f"Dataset might already exist: {e}")
    
    # Process data for each domain
    for domain, file_path in domain_paths.items():
        if not file_path.exists():
            print(f"Skipping {domain} due to missing file")
            continue
            
        print(f"\nProcessing domain: {domain}")
        
        # Load data for this domain
        domain_data = load_dataset_files([str(file_path)])
        
        # Ensure metadata contains the correct domain
        for entry in domain_data:
            if 'metadata' not in entry:
                entry['metadata'] = {}
            entry['metadata']['domain'] = domain
        
        # Split into train/test
        train_data, test_data = split_train_test(domain_data, domain_configs[domain]["train_ratio"])
        
        print(f"Uploading {len(train_data)} training examples and {len(test_data)} test examples")
        
        # Upload train split
        upload_domain_dataset(train_data, username, base_dataset_name, domain, 'train')
        
        # Add delay between train and test uploads
        time.sleep(5)
        
        # Upload test split
        upload_domain_dataset(test_data, username, base_dataset_name, domain, 'test')
        
        print(f"Completed uploading {domain} dataset")
        
        # Add delay between domains
        time.sleep(10)


if __name__ == "__main__":
    main()
