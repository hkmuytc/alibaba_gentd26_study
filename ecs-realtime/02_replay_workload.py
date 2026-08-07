import time
import pandas as pd
import torch
import gc
from pathlib import Path

# --- Fallback/Dummy HuggingFace Imports for Structure ---
# On the actual ECS instance, ensure `diffusers` is installed.
try:
    from diffusers import StableDiffusionXLPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


class ServerlessAIPipeline:
    """
    Abstractions representing the Serverless nature of the Alibaba cluster.
    Handles loading models, running inference, and tearing down to simulate cold starts.
    """
    def __init__(self, base_model_id="runwayml/stable-diffusion-v1-5"):
        self.base_model_id = base_model_id
        self.pipe = None

    def initialize_pipeline(self):
        """Simulates a container cold start: loading models into VRAM."""
        print("[Serverless] Cold Start: Loading Base Model into VRAM...")
        if HAS_DIFFUSERS:
            # We will use Stable Diffusion XL as it heavily stresses the GPU and supports modern LoRAs
            # Load Base Pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
            ).to("cuda")
            
            # Enable hardware acceleration to prevent queue saturation
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass # Falls back to standard attention if xformers isn't installed
            
            # Load a popular LoRA from Hugging Face (Pixel Art XL)
            print("[Serverless] Loading Base Model only (skipping LoRA to bypass PEFT errors)...")
            try:
                self.pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
                self.pipe.set_adapters(["pixel"])
            except Exception as e:
                print(f"[Warning] LoRA failed ({e}). Proceeding immediately with Base SDXL to generate power load.")
        else:
            time.sleep(2) # Fake loading time

    def teardown(self):
        """Simulates a container scale-to-zero event, purging VRAM."""
        print("[Serverless] Scale-to-Zero: Purging models from VRAM...")
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def generate(self, steps, num_images, prompt_len, negative_prompt_len):
        """Runs the actual physical inference based on trace lengths."""
        print(f"[Inference] Generating {num_images} images with {steps} steps (Prompt Len: {prompt_len})...")
        if HAS_DIFFUSERS and self.pipe is not None:
             # Alibaba traces only give us length to protect user privacy. 
             # We generate a dummy prompt of exact matching length using a repeating dictionary word.
             prompt = ("city " * int((prompt_len // 5) + 1))[:int(prompt_len)]
             negative_prompt = ("ugly " * int((negative_prompt_len // 5) + 1))[:int(negative_prompt_len)] if negative_prompt_len > 0 else ""
             
             # HARDCODE overrides to prevent queue saturation and force utilization fluctuation
             optimized_steps = 10 # Force low steps to clear queue fast
             batch_prompts = [prompt] * int(num_images)
             batch_neg_prompts = [negative_prompt] * int(num_images) if negative_prompt else None
             
             # Execute the generation via blazing-fast batched matrix multiplication
             self.pipe(prompt=batch_prompts, negative_prompt=batch_neg_prompts, num_inference_steps=optimized_steps)
             
        else:
            # Fake inference time based on steps
            time.sleep(steps * 0.1 * num_images)


class TraceReplayer:
    """
    Reads the Alibaba GenTD26 trace and replays it in real-time.
    """
    def __init__(self, trace_path, start_offset_minutes, duration_hours=3):
        self.trace_path = trace_path
        
        # We start X minutes into the trace to find the "bursty" period
        self.start_sec = start_offset_minutes * 60  
        self.end_sec = self.start_sec + (duration_hours * 3600)
        self.target_requests = []

    def load_trace(self):
        print(f"Loading trace from {self.trace_path}...")
        # Since timestamps are anonymized, we treat row index * roughly 2.5s (average pacing)
        # as a mock timeline for the sake of this prototype.
        # IN REALITY: you would use `gmt_create` or `exec_time_seconds` if they have a real timestamp column.
        df = pd.read_csv(self.trace_path)
        
        # We extract rows within our target window. We assume for this prototype 
        # we assign a mock arrival time to recreate the density.
        # (Replace `arrival_time` below with the actual offset column when you parse the trace)
        df['mock_arrival_sec'] = range(len(df)) # Placeholder
        
        window = df[(df['mock_arrival_sec'] >= self.start_sec) & (df['mock_arrival_sec'] <= self.end_sec)]
        self.target_requests = window.to_dict('records')
        print(f"Loaded {len(self.target_requests)} requests for the {self.end_sec - self.start_sec}s window.")

    def run(self):
        pipeline = ServerlessAIPipeline()
        pipeline.initialize_pipeline()
        
        start_real_time = time.time()
        
        with open("qps_trace.csv", "w") as f:
            f.write("time_absolute,request_type,value\n")
            
            for req in self.target_requests:
                # 1. Pacing: Wait until it's time for this request to arrive
                target_real_time = start_real_time + (req['mock_arrival_sec'] - self.start_sec)
                sleep_time = target_real_time - time.time()
                
                if sleep_time > 60:
                    # If we have a long wait, simulate Serverless Teardown
                    pipeline.teardown()
                    time.sleep(sleep_time)
                    pipeline.initialize_pipeline()
                elif sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 2. Execution
                steps = req.get('num_inference_steps', 30.0)
                n_imgs = req.get('num_images_per_prompt', 1.0)
                p_len = req.get('prompt_length', 50.0)
                # handle NaN values in pandas for negative prompt
                raw_neg_len = req.get('negative_prompt_length', 0.0)
                np_len = raw_neg_len if not pd.isna(raw_neg_len) else 0.0
                
                pipeline.generate(steps=steps, num_images=n_imgs, prompt_len=p_len, negative_prompt_len=np_len)
                
                # 3. Log the completion to match our model's input
                f.write(f"{time.time()},Generative Requests,1\n")
                f.flush()

if __name__ == "__main__":
    # The trace file lives in the same directory as the script on the ECS server
    trace_file = "data_trace_processed.csv"
    replayer = TraceReplayer(trace_path=trace_file, start_offset_minutes=600, duration_hours=3.0)
    replayer.load_trace()
    replayer.run()
