import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import gradio as gr
import os
from datetime import datetime
import gc

class LocalImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.current_model = None
        
        # Available models (you can add more)
        self.models = {
            "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
            "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
            "Anything V4": "andite/anything-v4.0",
            "Realistic Vision": "SG161222/Realistic_Vision_V2.0",
        }
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def load_model(self, model_name):
        """Load the selected model"""
        model_id = self.models.get(model_name, "runwayml/stable-diffusion-v1-5")
        
        if self.current_model == model_id and self.pipe is not None:
            return f"Model {model_name} already loaded!"
        
        # Clear previous model from memory
        if self.pipe is not None:
            del self.pipe
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        try:
            print(f"Loading model: {model_name}")
            
            # Determine data type and load settings based on device
            if self.device == "cuda":
                torch_dtype = torch.float16
                variant = "fp16"
            else:
                torch_dtype = torch.float32
                variant = None
            
            # Load the model with proper settings
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "safety_checker": None,
                "requires_safety_checker": False,
                "use_safetensors": True,
            }
            
            if variant:
                load_kwargs["variant"] = variant
            
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
            except:
                # Fallback without variant if it fails
                load_kwargs.pop("variant", None)
                load_kwargs.pop("use_safetensors", None)
                self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
            
            # Use DPM-Solver for faster generation
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory efficient features if available (but skip xformers)
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            
            # Enable CPU offloading for low memory systems
            if self.device == "cpu":
                if hasattr(self.pipe, "enable_model_cpu_offload"):
                    try:
                        self.pipe.enable_model_cpu_offload()
                    except:
                        pass
            
            self.current_model = model_id
            return f"✅ Model {model_name} loaded successfully!"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def generate_image(self, prompt, negative_prompt="", width=512, height=512, 
                      num_steps=20, guidance_scale=7.5, seed=-1):
        """Generate image from prompt"""
        
        if self.pipe is None:
            return None, "❌ Please load a model first!"
        
        try:
            # Set random seed if not specified
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            print(f"Generating image with prompt: '{prompt[:50]}...'")
            print(f"Settings: {width}x{height}, steps: {num_steps}, guidance: {guidance_scale}, seed: {seed}")
            
            # Generate image
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            image = result.images[0]
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}_seed_{seed}.png"
            save_path = os.path.join("generated_images", filename)
            
            # Create directory if it doesn't exist
            os.makedirs("generated_images", exist_ok=True)
            image.save(save_path)
            
            info = f"✅ Image generated successfully!\n"
            info += f"Seed: {seed}\n"
            info += f"Saved as: {filename}\n"
            info += f"Model: {self.current_model}"
            
            return image, info
            
        except Exception as e:
            error_msg = f"❌ Error generating image: {str(e)}"
            print(error_msg)
            return None, error_msg

def create_interface():
    """Create Gradio interface"""
    generator = LocalImageGenerator()
    
    # Load default model
    generator.load_model("Stable Diffusion 1.5")
    
    with gr.Blocks(title="Local AI Image Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Local AI Image Generator")
        gr.Markdown("Generate images using local Stable Diffusion models - no API keys required!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(generator.models.keys()),
                    value="Stable Diffusion 1.5",
                    label="🤖 Select Model",
                    info="Choose your preferred model"
                )
                
                load_btn = gr.Button("🔄 Load Model", variant="secondary")
                model_status = gr.Textbox(
                    value="Model loaded: Stable Diffusion 1.5",
                    label="Model Status",
                    interactive=False
                )
                
                # Prompt inputs
                prompt = gr.Textbox(
                    label="✨ Prompt",
                    placeholder="A beautiful sunset over mountains, highly detailed, 8k resolution...",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt (Optional)",
                    placeholder="blurry, low quality, distorted...",
                    lines=2
                )
                
                # Generation settings
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    num_steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale")
                
                seed = gr.Number(
                    value=-1,
                    label="🎲 Seed (-1 for random)",
                    info="Use same seed for reproducible results"
                )
                
                generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Textbox(
                    label="Generation Info",
                    lines=4,
                    interactive=False
                )
        
        # Example prompts
        gr.Markdown("## 💡 Example Prompts")
        example_prompts = [
            ["A majestic dragon flying over a medieval castle, fantasy art, detailed", "blurry, low quality"],
            ["Portrait of a beautiful woman with flowers in her hair, oil painting style", "ugly, distorted"],
            ["Futuristic cityscape at night with neon lights, cyberpunk style", "daytime, natural"],
            ["A cute robot in a garden with colorful flowers, cartoon style", "scary, dark, horror"],
            ["Mountain landscape with a lake reflection, golden hour lighting", "night, dark, indoor"]
        ]
        
        examples = gr.Examples(
            examples=example_prompts,
            inputs=[prompt, negative_prompt],
            label="Click to try these examples"
        )
        
        # Event handlers
        load_btn.click(
            fn=generator.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=generator.generate_image,
            inputs=[prompt, negative_prompt, width, height, num_steps, guidance_scale, seed],
            outputs=[output_image, generation_info]
        )
        
        # Tips and info
        with gr.Accordion("📚 Tips & Information", open=False):
            gr.Markdown("""
            ### 🎯 Prompt Tips:
            - Be specific and descriptive
            - Mention art style (e.g., "digital art", "oil painting", "photorealistic")
            - Include quality terms like "highly detailed", "8k resolution", "masterpiece"
            - Use commas to separate concepts
            
            ### ⚙️ Settings Guide:
            - **Steps**: More steps = better quality but slower (20-30 is usually good)
            - **Guidance Scale**: How closely to follow prompt (7-15 works well)
            - **Seed**: Same seed + same settings = same image
            
            ### 🚫 Negative Prompt Ideas:
            - Quality: "blurry, low quality, pixelated, distorted"
            - Unwanted elements: "text, watermark, signature"
            - Style: "cartoon" (if you want realistic), "photorealistic" (if you want artistic)
            
            ### 💾 Generated images are automatically saved in the 'generated_images' folder!
            """)
    
    return interface

# Alternative command-line interface
def cli_generate():
    """Command line interface for batch generation"""
    generator = LocalImageGenerator()
    
    print("🎨 Local Image Generator - CLI Mode")
    print("Loading default model...")
    generator.load_model("Stable Diffusion 1.5")
    
    while True:
        print("\n" + "="*50)
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # Optional settings
        try:
            steps = int(input("Steps (default 20): ") or 20)
            guidance = float(input("Guidance scale (default 7.5): ") or 7.5)
            width = int(input("Width (default 512): ") or 512)
            height = int(input("Height (default 512): ") or 512)
        except:
            steps, guidance, width, height = 20, 7.5, 512, 512
        
        print("\n🚀 Generating image...")
        image, info = generator.generate_image(
            prompt=prompt,
            num_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height
        )
        
        if image:
            print(info)
            image.show()  # Opens image in default viewer
        else:
            print(info)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI version
        cli_generate()
    else:
        # Run web interface
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",  # Allow access from other devices on network
            server_port=7860,
            share=False,  # Set to True to create public link
            inbrowser=True  # Auto-open browser
        )