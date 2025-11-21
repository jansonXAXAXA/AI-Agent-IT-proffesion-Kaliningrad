
import os
import time
import psutil
import sys
import gc
from typing import List, Dict, Optional
import llama_cpp
from llama_cpp import Llama
import warnings
import nltk
from nltk.tokenize import word_tokenize
warnings.filterwarnings("ignore")

try:
    nltk.download('punkt', quiet=True)
except ImportError as e:
    print(f"Критическа ошибка, нет доступа к NLTK: {e}")

class FixedGemmaEngine:  
    def __init__(self, model_path: str = "/kaggle/input/gemma-3-1b-it-q4-k-m/gguf/default/1/gemma-3-1b-it-Q4_K_M.gguf"):
        self.model_path = model_path
        self.n_threads = min(4, max(1, psutil.cpu_count(logical=False) or 4))
        self.max_context_size = 2048     
        self.model = self._load_model()
    
    def _load_model(self) -> Llama:        
        try:
            start_time = time.time()
            
            # CRITICAL FIXES FOR GEMMA MODELS:
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
                chat_format="gemma",
            )
            
            load_time = time.time() - start_time
            return model
            
        except Exception as e:
            print(f"FATAL MODEL LOADING ERROR: {e}")
            sys.exit(1)
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                stop_tokens: Optional[List[str]] = ["<|eot_id|>", "</s>", "<eos>", "<|end_of_text|>"]) -> str:
        if stop_tokens is None:
            stop_tokens = ["<|eot_id|>", "</s>", "<eos>", "<|end_of_text|>", "<|end_of_turn|>"]
        
        try:
            start_time = time.time()
            
            if not prompt.startswith("<start_of_turn>") and not prompt.startswith("user:"):
                prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            output = self.model(
                prompt=prompt,
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
            result = raw_text.strip()
            
            generation_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"Ошибка генерации, включен безопасный режим: {e}")
            return self._fallback_generation(prompt, max_tokens, temperature)
    
    def _fallback_generation(self, prompt: str, max_tokens: int = 64, temperature: float = 0.3) -> str:
        try:           
            output = self.model(
                prompt=prompt[:128],
                max_tokens=min(max_tokens, 32),
                temperature=temperature,
                top_p=0.9,
                top_k=30,
                stop=["<|eot_id|>", "</s>"],
                echo=False,
                stream=False
            )
            
            result = output['choices'][0]['text'].strip()
            return result
            
        except Exception as e:
            return f"Критическая ошибка: {str(e)}"
    
    def cleanup_memory(self):   
        if hasattr(self, 'cache'):
            self.cache.clear()
        
        gc.collect()
    
    def shutdown(self):
        self.cleanup_memory()
        
        if hasattr(self, 'model') and self.model is not None:
            try:
                mem_before = psutil.virtual_memory().used
                
                del self.model
                
                gc.collect()
                
                mem_after = psutil.virtual_memory().used
                mem_freed = (mem_before - mem_after) / 1024**3
                
                self.model = None
                
            except Exception as e:
                print(f"Критическая ошибка (очистка): {e}")
        
        gc.collect()
        gc.collect()
        
        mem = psutil.virtual_memory()

def LLM_FUNCTION(query): 
    try:
        engine = FixedGemmaEngine()

        prompt = input()
        result = engine.generate(
            prompt,
            max_tokens=256,
            temperature=0.7
        )

        try:
            engine.shutdown()
        except Exception as e:
            print(f"Критическая ошибка при выключении движка: {e}")
        
        return result
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None


def GET_LLM_ANSWER():
    try:
        result = LLM_FUNCTION()
            
        gc.collect()
        gc.collect()
            
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

        return result
                
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)
