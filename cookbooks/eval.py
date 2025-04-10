from unsloth import FastLanguageModel
import torch
import json
from tqdm import tqdm
import pandas as pd
from transformers import TextStreamer
from typing import Dict, List
import numpy as np
import asyncio
from camel.extractors import BaseExtractor, BoxedStrategy
from datasets import load_dataset
import os
from huggingface_hub import login
from camel.logger import get_logger, set_log_level

# Set up CAMEL logger
logger = get_logger(__name__)
set_log_level('INFO')

class ModelEvaluator:
    def __init__(self, hf_token: str = None):
        """
        Initialize the evaluator with optional Hugging Face token
        """
        self.hf_token = hf_token
        if hf_token:
            login(token=hf_token)
        self.extractor = None
        self.base_model = None
        self.base_tokenizer = None
        self.ft_model = None
        self.ft_tokenizer = None
        
    async def setup(self):
        """
        Set up the extractor and models
        """
        await self.setup_extractor()
        self.load_models()
        
    async def setup_extractor(self):
        """
        Set up the CAMEL extractor with BoxedStrategy
        """
        self.extractor = BaseExtractor([[BoxedStrategy()]])
        await self.extractor.setup()
        logger.info("Extractor setup complete")

    def load_models(self):
        """
        Load the base and fine-tuned models
        """
        logger.info("Loading base model...")
        try:
            self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(
                model_name="Qwen/Qwen2.5-7B",
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self.base_model)
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
        
        logger.info("Loading fine-tuned model...")
        try:
            self.ft_model, self.ft_tokenizer = FastLanguageModel.from_pretrained(
                model_name="fintuned_model_name",
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self.ft_model)
            logger.info("Fine-tuned model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def extract_boxed_answer(self, text: str) -> str:
        """
        Extract answer from boxed notation in the text
        Returns cleaned answer or original text if no boxed answer found
        """
        try:
            extractions = self.extractor.extract(text)
            if extractions and len(extractions) > 0:
                return extractions[0].strip().lower()
        except Exception as e:
            logger.error(f"Extraction error: {e}")
        return text.strip().lower()

    def calculate_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[float]:
        """
        Calculate accuracy metrics comparing predictions against ground truths using the extractor
        """
        correct = 0
        total = len(ground_truths)
        
        for pred, truth in zip(predictions, ground_truths):
            cleaned_pred = self.extract_boxed_answer(pred)
            cleaned_truth = self.extract_boxed_answer(truth)
            
            if cleaned_pred == cleaned_truth:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
        """
        Generate response from the model
        """
        try:
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            text_streamer = TextStreamer(tokenizer)
            
            outputs = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("<|assistant|>\n")[-1].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

    def load_test_data(self, dataset_name: str, split: str = "test"):
        """
        Load test data from Hugging Face dataset
        """
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, split=split)
            logger.info(f"Successfully loaded {len(dataset)} examples from dataset")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    async def evaluate(self, dataset_name: str, num_samples: int = None):
        """
        Main evaluation function
        """
        # Load test data
        test_data = self.load_test_data(dataset_name)
        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))
            logger.info(f"Selected {num_samples} samples for evaluation")

        results = []
        base_predictions = []
        ft_predictions = []
        ground_truths = []
        
        logger.info("Starting evaluation...")
        for idx, example in enumerate(tqdm(test_data)):
            question = example['question']
            final_answer = example['final_answer']
            
            logger.info(f"\nEvaluating example {idx + 1}/{len(test_data)}")
            logger.info(f"Question: {question}")
            logger.info(f"Ground Truth: {final_answer}")
            
            # Generate responses
            logger.info("Generating base model response...")
            base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
            logger.info(f"Base Model Answer: {base_response}")
            
            logger.info("Generating fine-tuned model response...")
            ft_response = self.generate_response(self.ft_model, self.ft_tokenizer, question)
            logger.info(f"Fine-tuned Model Answer: {ft_response}")
            
            # Extract boxed answers
            base_extracted = self.extract_boxed_answer(base_response)
            ft_extracted = self.extract_boxed_answer(ft_response)
            truth_extracted = self.extract_boxed_answer(final_answer)
            
            results.append({
                'question': question,
                'ground_truth': final_answer,
                'ground_truth_extracted': truth_extracted,
                'base_model_answer': base_response,
                'base_model_extracted': base_extracted,
                'fine_tuned_answer': ft_response,
                'fine_tuned_extracted': ft_extracted,
                'base_correct': base_extracted == truth_extracted,
                'ft_correct': ft_extracted == truth_extracted
            })
            
            base_predictions.append(base_response)
            ft_predictions.append(ft_response)
            ground_truths.append(final_answer)
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(base_predictions, ground_truths)
        ft_metrics = self.calculate_metrics(ft_predictions, ground_truths)
        
        # Log results
        logger.info("\n=== Evaluation Results ===")
        logger.info("\nBase Model Metrics:")
        logger.info(f"Accuracy: {base_metrics['accuracy']:.2%}")
        logger.info(f"Correct: {base_metrics['correct']}/{base_metrics['total']}")
        
        logger.info("\nFine-tuned Model Metrics:")
        logger.info(f"Accuracy: {ft_metrics['accuracy']:.2%}")
        logger.info(f"Correct: {ft_metrics['correct']}/{ft_metrics['total']}")
        
        # Save results
        self.save_results(results, base_metrics, ft_metrics)

    def save_results(self, results: List[Dict], base_metrics: Dict, ft_metrics: Dict):
        """
        Save evaluation results to files
        """
        try:
            # Save detailed results to CSV
            df = pd.DataFrame(results)
            df.to_csv('evaluation_results.csv', index=False)
            logger.info("Detailed results saved to evaluation_results.csv")
            
            # Save summary metrics
            summary = {
                'base_model': base_metrics,
                'fine_tuned_model': ft_metrics,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            with open('evaluation_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info("Summary metrics saved to evaluation_summary.json")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

async def main():
    # Check for HF token in environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("No Hugging Face token found in environment variables")
        hf_token = "your_hf_token_here"  
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(hf_token=hf_token)
        
        # Setup evaluator
        await evaluator.setup()
        
        # Run evaluation
        await evaluator.evaluate(
            dataset_name="EleutherAI/hendrycks_math",  
            num_samples = None
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())