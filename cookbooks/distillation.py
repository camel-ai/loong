import asyncio
import json
import os
import time
from multiprocessing import Pool, freeze_support

from camel.datasets.static_dataset import StaticDataset
from camel.datasets.few_shot_generator import FewShotGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.extractors import BaseExtractor, BoxedStrategy
from camel.verifiers import MathVerifier, PythonVerifier
from camel.environments import SingleStepEnv, Action
from camel.logger import get_logger, set_log_level

# Logger setup
logger = get_logger(__name__)
set_log_level('INFO')

# Function to process a single record
def process_record(process_id, user_prompt, api_key, seed_file_path, output_file):
    # Create a new environment with all necessary components for this process
    try:
        # Set up DeepSeek environment variable
        os.environ["GET_REASONING_CONTENT"] = "true"
        
        # Initialize model for this process
        model_process = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_REASONER,
            api_key=api_key,
            timeout=1000
        )
        
        # Initialize agent
        local_agent = ChatAgent(model=model_process)
        
        # Set up environment components
        extractor = BaseExtractor([[BoxedStrategy()]])
        asyncio.run(extractor.setup())
        
        # Python verifier
        python_verifier = PythonVerifier(required_packages=["sympy"])
        asyncio.run(python_verifier.setup(uv=True))
        
        # Math verifier
        math_verifier = MathVerifier(
            extractor=extractor,
            float_rounding=6,
            numeric_precision=15,
            enable_wrapping=True
        )
        asyncio.run(math_verifier.setup())
        
        # Load seed dataset
        with open(seed_file_path, 'r') as f:
            seed_data = json.load(f)
        
        seed_dataset = StaticDataset(seed_data)
        
        # Initialize generator 
        model_4o = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict=ChatGPTConfig().as_dict(),
            timeout=1000
        )
        
        generator = FewShotGenerator(
            buffer=10,
            seed_dataset=seed_dataset,
            verifier=python_verifier,
            model=model_4o
        )
        
        # Create environment
        env = SingleStepEnv(generator, math_verifier)
        asyncio.run(env.setup())
        
        # Reset environment and get question
        obs = asyncio.run(env.reset())
        question = obs.question
        
        # This is the bottleneck operation we're parallelizing
        deepseek_response = local_agent.step(user_prompt + question).msgs[0].content
        
        # Split the response into reasoning and answer parts
        reasoning_part = ""
        answer_part = deepseek_response
        
        if "<think>" in deepseek_response and "</think>" in deepseek_response:
            parts = deepseek_response.split("</think>")
            if len(parts) > 1:
                reasoning_part = parts[0].replace("<think>", "").strip()
                answer_part = parts[1].strip()
        
        # Verify the result
        next_obs, reward, done, info = asyncio.run(env.step(Action(index=0, llm_response=deepseek_response)))
        
        # Create data entry
        data_entry = {
            "question": question,
            "answer": info['state'].final_answer if 'state' in info else '',
            "response": answer_part,
            "long_cot": reasoning_part,
            "shots": obs.metadata.get('shots'),
            "verified": reward > 0
        }
        
        # Return the result - we'll handle shared state synchronization in the main process
        return data_entry, reward > 0, process_id
    
    except Exception as e:
        logger.error(f"Error processing record {process_id}: {str(e)}")
        return None, False, process_id

def main():
    start_time = time.time()
    
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("No OpenAI API key found")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("No DeepSeek API key found")
        
    # Setup file paths
    OUTPUT_FILE = "math_dataset.json"
    SEED_FILE_PATH = '/Users/enrei/Desktop/camel0209/camel/camel/verifiers/seed_dataset_first_20.json'
    BATCH_SIZE = 50

    # Load existing dataset if it exists
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            dataset = json.load(f)
        logger.info(f"Loaded existing dataset with {len(dataset)} examples")
    else:
        dataset = []
        logger.info("Starting new dataset")

    logger.info("Loading advanced math dataset...")
    # Load the seed dataset
    with open(SEED_FILE_PATH, 'r') as f:
        seed_data = json.load(f)

    logger.info(f"Loaded seed dataset with {len(seed_data)} examples")
    
    # Define the prompt for CoT generation
    USER_PROMPT = """You are an agent designed to answer mathematical questions with clarity and precision. Your task is to provide a step-by-step explanation for
any mathematical problem posed by the user, ensuring the response is easy to follow. Adhere to these guidelines:
Analyze the mathematical question carefully and break down the solution process into clear, logical steps.
Use natural language to explain each step, incorporating LaTeX notation (e.g., $x + 2$)
for mathematical expressions when helpful. Conclude your response with the final answer enclosed
in a LaTeX \boxed{} environment (e.g., \boxed{5}).
Place this at the end of your explanation as a standalone statement.
It should be a Python expression, for example "[1, 2, 3]" for a list.

The question you should answer is: """

    num_rejected = 0
    target_size = 1000
    verified_count = sum(1 for entry in dataset if entry["verified"])
    
    logger.info("Starting generation and verification loop...")

    # Main processing loop with proper process pool management
    while verified_count < target_size:
        logger.info(f"Current verified count: {verified_count}/{target_size}")
        
        # Determine batch size
        remaining = target_size - verified_count
        batch_size = min(BATCH_SIZE, remaining)
        
        # Create arguments for each process
        process_args = [
            (i, USER_PROMPT, os.environ.get("DEEPSEEK_API_KEY"), SEED_FILE_PATH, OUTPUT_FILE) 
            for i in range(batch_size)
        ]
        
        logger.info(f"Processing batch with {batch_size} processes...")
        with Pool(processes=batch_size) as pool:
            # Use starmap to pass multiple arguments to process_record
            results = pool.starmap(process_record, process_args)
            
            # Close and join pool properly
            pool.close()
            pool.join()
            
            # Process results from completed processes
            newly_verified = 0
            for data_entry, is_verified, proc_id in results:
                if data_entry is not None:
                    # Add to dataset
                    dataset.append(data_entry)
                    
                    # Update counters
                    if is_verified:
                        newly_verified += 1
                        verified_count += 1
                        logger.info(f"Process {proc_id}: Verification successful - Added verified entry ({verified_count}/{target_size} verified)")
                    else:
                        num_rejected += 1
                        logger.warning(f"Process {proc_id}: Verification failed - Added unverified entry ({verified_count}/{target_size} verified)")
            
            # Save after each batch
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(dataset, f, indent=2)
                
            logger.info(f"Batch complete - Added {newly_verified} verified entries in this batch")
    
    # Final statistics
    total_entries = len(dataset)
    verified_entries = sum(1 for entry in dataset if entry["verified"])
    logger.info(f"Generation complete. Total entries: {total_entries}")
    logger.info(f"Verified entries: {verified_entries}")
    logger.info(f"Rejected entries: {num_rejected}")

    end_time = time.time()

    # Calculate total elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Total elapsed time: {elapsed_time:.2f}s")
    logger.info(f"Average time per record: {elapsed_time/total_entries:.2f}s")
    logger.info(f"Processed using up to {num_processes} worker processes.")

# Required for multiprocessing on macOS (and Windows)
if __name__ == '__main__':
    # Add freeze_support to properly handle multiprocessing in frozen executables
    freeze_support()
    main()
