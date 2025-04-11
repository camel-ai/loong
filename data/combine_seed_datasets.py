#!/usr/bin/env python3
"""
Script to combine seed datasets from all domains into a single JSON file.
This script reads seed_dataset.json files from each domain directory
and combines them into a single JSON file named 'seed_dataset_all_domain.json'.
"""

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def main():
    # Define the base data directory
    data_dir = Path(__file__).parent

    # Define the domains to process with their paths
    domain_paths = {
        "advanced_math": data_dir / "advanced_math" / "seed_dataset.json",
        "advanced_physics": data_dir / "advanced_physics" / "seed_dataset.json",
        "computational_biology": data_dir / "computational_biology" / "seed_dataset.json",
        "finance": data_dir / "finance" / "seed_dataset.json",
        "games": data_dir / "games" / "blackjack" / "seed_dataset.json",  # Special case for games
        "graph_discrete_math": data_dir / "graph_discrete_math" / "seed_dataset.json",
        "logic": data_dir / "logic" / "seed_dataset.json",
        "mathematical_programming": data_dir / "mathematical_programming" / "seed_dataset.json",
        "security_and_safety": data_dir / "security_and_safety" / "seed_dataset.json"
    }
    # Dictionary to hold all domain data
    all_domains_data = {}

    # Process each domain
    for domain, domain_path in domain_paths.items():
        if domain_path.exists():
            logger.info(f"Processing {domain}...")
            try:
                with open(domain_path, 'r', encoding='utf-8') as f:
                    domain_data = json.load(f)
                    all_domains_data[domain] = domain_data
                logger.info(f"Successfully loaded {domain} with {len(domain_data)} entries")
            except Exception as e:
                logger.error(f"Error loading {domain}: {e}")
        else:
            logger.warning(f"Warning: {domain_path} does not exist")
    
    # Write the combined data to a new file
    output_file = data_dir / "seed_dataset_all_domain.json"
    
    logger.info(f"\nWriting combined data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_domains_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    total_domains = len(all_domains_data)
    logger.info(f"\nCombined {total_domains} domains into {output_file}")
    logger.info("Domains included:")
    for domain in all_domains_data:
        logger.info(f"- {domain}: {len(all_domains_data[domain])} entries")

if __name__ == "__main__":
    main()
