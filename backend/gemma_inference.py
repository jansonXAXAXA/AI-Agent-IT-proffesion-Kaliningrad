import os
import time
import psutil
import sys
import gc
import logging
from typing import List, Dict, Optional
import llama_cpp
from llama_cpp import Llama
import warnings
from config import config
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("FixedGemmaEngine")
logger.propagate = False

logging.getLogger("llama_cpp").setLevel(logging.WARNING)

MODEL_PATH = config.MODEL_PATH

class FixedGemmaEngine:  
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.n_threads = min(4, max(1, psutil.cpu_count(logical=False) or 4))
        self.max_context_size = 8192  
        self.model = self._load_model()
    
    def _load_model(self) -> Llama:        
        try:
            start_time = time.time()
            logger.info(f"Loading model from {self.model_path} with {self.n_threads} threads...")
            
            model = Llama(
                model_path=self.model_path,
                n_ctx=self.max_context_size,
                n_threads=self.n_threads,
                n_threads_batch=self.n_threads,
                n_gpu_layers=0,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                logits_all=False,
                embedding=False,
                low_vram=False,
                chat_format="gemma"
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return model
            
        except Exception as e:
            logger.critical(f"FATAL MODEL LOADING ERROR: {e}", exc_info=True)
            sys.exit(1)
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                stop_tokens: Optional[List[str]] = None) -> str:
        if stop_tokens is None:
            stop_tokens = ["<end_of_turn>", "<eos>", "<|end_of_turn|>", "<|end_of_text|>"]
        
        try:
            logger.debug(f"Generating response for prompt: {prompt[:50]}...")
            formatted_prompt = (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            
            output = self.model(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                repeat_penalty=1.0,
                seed=int(time.time()) % 10000,
                stop=stop_tokens,
                echo=False,
                stream=False
            )
            
            raw_text = output['choices'][0]['text']
            for token in stop_tokens:
                if raw_text.endswith(token):
                    raw_text = raw_text[:-len(token)]
            result = raw_text.strip()
            logger.debug(f"Generated response: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.warning(f"Generation error, using fallback: {e}", exc_info=True)
            return self._fallback_generation(prompt, max_tokens, temperature)
    
    def _fallback_generation(self, prompt: str, max_tokens: int = 64, temperature: float = 0.3) -> str:
        try:           
            logger.warning("Using fallback generation parameters")
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            output = self.model(
                prompt=formatted_prompt,
                max_tokens=min(max_tokens, 32),
                temperature=temperature,
                top_p=0.9,
                top_k=30,
                stop=["<end_of_turn>", "<eos>"],
                echo=False,
                stream=False
            )
            
            result = output['choices'][0]['text'].strip()
            logger.debug(f"Fallback result: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.critical(f"Fallback generation failed: {e}", exc_info=True)
            return f"CRITICAL ERROR: {str(e)}"
    
    def cleanup_memory(self):   
        logger.debug("Performing garbage collection")
        gc.collect()
    
    def shutdown(self):
        logger.info("Shutting down engine...")
        self.cleanup_memory()
        
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                self.model = None
                gc.collect()
                logger.info("Model resources released")
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)
        
        gc.collect()
        logger.info("Engine shutdown complete")

def LLM_FUNCTION(query): 
    try:
        logger.info(f"Processing query: {query[:50]}...")
        engine = FixedGemmaEngine()
        result = engine.generate(
            query,
            max_tokens=256,
            temperature=0.7
        )
        engine.shutdown()
        logger.info(f"Query processed successfully")
        return result
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in LLM_FUNCTION: {e}", exc_info=True)
        return None

def GET_LLM_ANSWER(query):
    try:
        logger.info("=== LLM PROCESSING ===")
        result = LLM_FUNCTION(query)
        gc.collect()
        logger.info("=== PROCESSING COMPLETE ===")
        return result
                
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in GET_LLM_ANSWER: {e}", exc_info=True)
        sys.exit(1)
