#!/usr/bin/env python3
import json
from pathlib import Path
import asyncio
import logging
from typing import Dict, List, Any
from tqdm import tqdm
from camel.verifiers.python_verifier import PythonVerifier
from camel.verifiers.models import VerificationOutcome
from camel.verifiers import MathVerifier
from physic_verifier_tem import PhysicsVerifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

async def setup_verifier(required_packages: List[str], timeout: float = 60.0) -> PythonVerifier:
    """Setup a Python verifier with required packages"""
    logger.info(f"Setting up verifier with packages: {required_packages}")
    verifier = PythonVerifier(timeout=timeout, required_packages=required_packages)
    await verifier.setup(uv=True)
    return verifier

async def execute_rationale(rationale: str, verifier: PythonVerifier) -> Dict[str, Any]:
    """Execute a single rationale using the verifier"""
    try:
        result = await verifier.verify(rationale, None)
        return {
            "status": result.status.name,
            "result": result.result,
            "error_message": result.error_message,
            "execution_successful": result.status == VerificationOutcome.SUCCESS
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "result": "",
            "error_message": str(e),
            "execution_successful": False
        }

async def compare_results(execution_result: str, final_answer: str, domain: str = None) -> bool:
    """Compare execution results with expected answer, using domain-specific verifiers if needed"""
    if domain == "advanced_math":
        try:
            logger.info("Using MathVerifier for comparison")
            math_verifier = MathVerifier(float_rounding=6, numeric_precision=15)
            await math_verifier.setup()
            verification_result = await math_verifier.verify(
                solution=execution_result,
                reference_answer=final_answer
            )
            return verification_result.status == VerificationOutcome.SUCCESS
        except Exception as e:
            logger.warning(f"MathVerifier failed: {e}")
            pass
    elif domain == "advanced_physics":
        try:
            logger.info("Using PhysicsVerifier for comparison")
            physics_verifier = PhysicsVerifier(float_rounding=6, numeric_precision=15)
            await physics_verifier.setup(uv=True)
            verification_result = await physics_verifier.verify(
                solution=execution_result,
                reference_answer=final_answer
            )
            return verification_result.status == VerificationOutcome.SUCCESS
        except Exception as e:
            logger.warning(f"PhysicsVerifier failed: {e}")
            pass
    
    return execution_result == final_answer

async def validate_domain_data(domain: str, items: List[Dict], verifier: PythonVerifier) -> Dict[str, float]:
    """Validate all items in a domain"""
    successful_executions = 0
    successful_matches = 0
    total_items = len(items)
    
    logger.info(f"\nProcessing {total_items} items in {domain}")
    
    # Use tqdm for progress bar
    for i, item in enumerate(tqdm(items, desc=f"Validating {domain}", unit="item")):
        rationale = item.get("rationale", "")
        final_answer = item.get("final_answer", "")

        if not rationale or not final_answer:
            logger.warning(f"Skipping item {i}: Missing rationale or final_answer")
            continue

        execution_output = await execute_rationale(rationale, verifier)
        
        if execution_output["execution_successful"]:
            successful_executions += 1
            match_status = await compare_results(execution_output["result"], final_answer, domain)
            if match_status:
                successful_matches += 1
            else:
                # Log mismatch details for debugging
                logger.debug(f"Mismatch in item {i}:")
                logger.debug(f"Expected: {final_answer}")
                logger.debug(f"Got: {execution_output['result']}")

        # Show intermediate progress every 10 items
        if (i + 1) % 10 == 0:
            current_execution_rate = (successful_executions / (i + 1)) * 100
            current_match_rate = (successful_matches / (i + 1)) * 100
            logger.info(f"Intermediate results ({i + 1}/{total_items}):")
            logger.info(f"Current Execution Rate: {current_execution_rate:.2f}%")
            logger.info(f"Current Match Rate: {current_match_rate:.2f}%")

    return {
        "total": total_items,
        "execution_rate": (successful_executions / total_items) * 100 if total_items > 0 else 0,
        "match_rate": (successful_matches / total_items) * 100 if total_items > 0 else 0
    }

async def main():
    """Main function to validate all domains"""
    data_dir = Path(__file__).parent
    # Define paths for all domain datasets
    domain_paths = {
        "advanced_math": data_dir / "advanced_math" / "seed_dataset.json",
        "advanced_physics": data_dir / "advanced_physics" / "seed_dataset.json",
        "computational_biology": data_dir / "computational_biology" / "seed_dataset.json",
        "finance": data_dir / "finance" / "seed_dataset.json",
        "games": data_dir / "games" / "blackjack" / "seed_dataset.json",
        "graph_discrete_math": data_dir / "graph_discrete_math" / "seed_dataset.json",
        # "logic": data_dir / "logic" / "seed_dataset.json",
        "mathematical_programming": data_dir / "mathematical_programming" / "seed_dataset.json",
        # "security_and_safety": data_dir / "security_and_safety" / "seed_dataset.json"
    }

    results = {}
    all_success = True
    total_domains = len(domain_paths)
    processed_domains = 0

    logger.info(f"Starting validation of {total_domains} domains")
    
    for domain, path in domain_paths.items():
        processed_domains += 1
        logger.info(f"\n[{processed_domains}/{total_domains}] Processing domain: {domain}")
        
        if not path.exists():
            logger.error(f"❌ {domain}: Dataset file not found at {path}")
            all_success = False
            continue

        try:
            logger.info(f"Loading dataset for {domain}...")
            with open(path, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            logger.info(f"Loaded {len(domain_data)} items from {domain}")
        except Exception as e:
            logger.error(f"❌ {domain}: Failed to load dataset - {e}")
            all_success = False
            continue

        # Setup verifier with required packages
        required_packages = ["numpy", "pandas", "sympy"]  # Adjust as needed
        logger.info(f"Setting up verifier for {domain}...")
        verifier = await setup_verifier(required_packages)

        try:
            stats = await validate_domain_data(domain, domain_data, verifier)
            results[domain] = stats
            
            # Check if domain achieved 100% success
            if stats["execution_rate"] < 100 or stats["match_rate"] < 100:
                all_success = False
                logger.error(f"❌ {domain}:")
            else:
                logger.info(f"✅ {domain}:")
            
            logger.info(f"   Total Items: {stats['total']}")
            logger.info(f"   Execution Success Rate: {stats['execution_rate']:.2f}%")
            logger.info(f"   Match Rate: {stats['match_rate']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ {domain}: Validation failed - {e}")
            all_success = False
        finally:
            logger.info(f"Cleaning up verifier for {domain}")
            await verifier.cleanup()

    # Final summary
    logger.info("\n=== Final Summary ===")
    for domain, stats in results.items():
        status = "✅" if stats["execution_rate"] == 100 and stats["match_rate"] == 100 else "❌"
        logger.info(f"{status} {domain}: Execution {stats['execution_rate']:.2f}% | Match {stats['match_rate']:.2f}%")

    if not all_success:
        logger.error("\n❌ Validation failed: Not all domains achieved 100% success rate")
        exit(1)
    else:
        logger.info("\n✅ All domains validated successfully")
        exit(0)

if __name__ == "__main__":
    asyncio.run(main())
