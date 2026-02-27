import gradio as gr
import os
from datetime import datetime
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv
import random

# Load token from .env file
load_dotenv()

class HFImageGenerator:
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN")
        self.client = None

        # Models running on HF Inference API (no local download)
        self.models = {
            "Stable Diffusion XL":   "stabilityai/stable-diffusion-xl-base-1.0",
            "Stable Diffusion 2.1":  "stabilityai/stable-diffusion-2-1",
            "Stable Diffusion 1.5":  "runwayml/stable-diffusion-v1-5",
            "Realistic Vision V4":   "SG161222/RealVisXL_V4.0",
            "DreamShaper XL":        "Lykon/dreamshaper-xl-1-0",
            "Playground v2.5":       "playgroundai/playground-v2.5-1024px-aesthetic",
        }

        # Auto-authenticate if token found in .env
        if self.hf_token:
            try:
                self.client = InferenceClient(token=self.hf_token)
                print("✅ HF token loaded from .env and authenticated successfully!")
            except Exception as e:
                print(f"❌ Token found but authentication failed: {e}")
        else:
            print("⚠️  No HF_TOKEN found in .env file. Please add it and restart.")

    def get_token_status(self):
        if self.client and self.hf_token:
            return "✅ Authenticated via .env file — ready to generate!"
        return "❌ HF_TOKEN not found in .env file. Please check your .env and restart."

    def generate_image(self, prompt, negative_prompt, width, height,
                       num_steps, guidance_scale, seed, model_name):
        """Generate image via HF Inference API"""

        if not self.client:
            return None, "❌ Not authenticated. Add HF_TOKEN to your .env file and restart the app."

        if not prompt.strip():
            return None, "❌ Please enter a prompt!"

        model_id = self.models.get(model_name, "stabilityai/stable-diffusion-xl-base-1.0")
        actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)

        try:
            print(f"Generating with model : {model_id}")
            print(f"Prompt               : {prompt[:80]}...")
            print(f"Settings             : {width}x{height}, steps={num_steps}, guidance={guidance_scale}, seed={actual_seed}")

            gen_kwargs = {
                "width": width,
                "height": height,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
            }
            if negative_prompt.strip():
                gen_kwargs["negative_prompt"] = negative_prompt.strip()

            image = self.client.text_to_image(
                prompt=prompt,
                model=model_id,
                **gen_kwargs
            )

            # Save image
            os.makedirs("generated_images", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}_seed_{actual_seed}.png"
            save_path = os.path.join("generated_images", filename)
            image.save(save_path)

            info = (
                f"✅ Image generated successfully!\n"
                f"Model : {model_name}\n"
                f"Seed  : {actual_seed}\n"
                f"Size  : {width}x{height}\n"
                f"Saved : {filename}"
            )
            return image, info

        except Exception as e:
            error = str(e)
            if "401" in error or "unauthorized" in error.lower():
                msg = "❌ Unauthorized — your HF token may be invalid or expired. Update .env and restart."
            elif "503" in error or "loading" in error.lower():
                msg = "⏳ Model is loading on HF servers. Wait 20–30 seconds and try again."
            elif "429" in error or "rate" in error.lower():
                msg = "⚠️ Rate limit reached. Wait a moment or upgrade to HF Pro."
            elif "400" in error:
                msg = f"❌ Bad request — try different dimensions or fewer steps.\nDetails: {error}"
            else:
                msg = f"❌ Error: {error}"
            print(msg)
            return None, msg


def create_interface():
    generator = HFImageGenerator()

    with gr.Blocks(title="HF AI Image Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Hugging Face AI Image Generator")
        gr.Markdown("Generate images using the Hugging Face Inference API — token loaded from `.env` file.")

        # Token status banner
        token_status = gr.Textbox(
            label="🔑 Token Status",
            value=generator.get_token_status(),
            interactive=False
        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):

                model_dropdown = gr.Dropdown(
                    choices=list(generator.models.keys()),
                    value="Stable Diffusion XL",
                    label="🤖 Select Model",
                    info="All models run on HF servers — no local download needed"
                )

                prompt = gr.Textbox(
                    label="✨ Prompt",
                    placeholder="A beautiful sunset over mountains, highly detailed, 8k resolution...",
                    lines=3
                )

                negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt (Optional)",
                    placeholder="blurry, low quality, distorted, watermark...",
                    lines=2
                )

                with gr.Row():
                    width  = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")

                with gr.Row():
                    num_steps      = gr.Slider(10, 50, value=25, step=1,      label="Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale")

                seed = gr.Number(
                    value=-1,
                    label="🎲 Seed (-1 for random)",
                    info="Fix a seed to reproduce the same image"
                )

                generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_image    = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Textbox(label="Generation Info", lines=5, interactive=False)

        # Example prompts
        gr.Markdown("## 💡 Example Prompts")
        gr.Examples(
            examples=[
                ["A majestic dragon flying over a medieval castle, fantasy art, highly detailed", "blurry, low quality"],
                ["Portrait of a woman with flowers in her hair, oil painting style, masterpiece",  "ugly, distorted"],
                ["Futuristic cityscape at night with neon lights, cyberpunk style, 8k",            "daytime, natural lighting"],
                ["A cute robot in a garden with colorful flowers, digital art",                    "scary, dark, horror"],
                ["Mountain landscape with lake reflection, golden hour lighting, photorealistic",  "night, dark, indoor"],
                ["Astronaut riding a horse on Mars, cinematic lighting, ultra detailed",           "low quality, blurry"],
            ],
            inputs=[prompt, negative_prompt],
            label="Click to use these examples"
        )

        with gr.Accordion("📚 Tips & Information", open=False):
            gr.Markdown("""
            ### 🔑 .env Setup:
            Create a file named `.env` in the same folder as this script:
            ```
            HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            ```
            Then restart the app — the token loads automatically.
            Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **read** scope.

            ### 🎯 Prompt Tips:
            - Be specific and descriptive
            - Mention art style: "digital art", "oil painting", "photorealistic"
            - Add quality boosters: "highly detailed", "8k resolution", "masterpiece"

            ### ⚙️ Settings Guide:
            - **Steps**: 20–30 is a good balance of quality vs speed
            - **Guidance Scale**: 7–12 works well; higher = more literal prompt following
            - **Seed**: Use -1 for random, or fix a seed to reproduce results

            ### 💾 Images are auto-saved in the `generated_images/` folder.

            ### ⚠️ HF Free Tier Limits:
            - Free accounts have rate limits — wait a few seconds between generations
            - Some models may take 20–30s to load (cold start) — just retry
            - Upgrade to [HF Pro](https://huggingface.co/pricing) for priority access
            """)

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[prompt, negative_prompt, width, height, num_steps, guidance_scale, seed, model_dropdown],
            outputs=[output_image, generation_info]
        )

    return interface


def cli_generate():
    """Command line interface"""
    generator = HFImageGenerator()
    print(generator.get_token_status())

    if not generator.client:
        print("Add HF_TOKEN to your .env file and restart.")
        return

    while True:
        print("\n" + "=" * 50)
        prompt = input("Enter prompt (or 'quit' to exit): ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break

        model_name = input("Model (default: Stable Diffusion XL): ").strip() or "Stable Diffusion XL"

        try:
            steps    = int(input("Steps (default 25): ") or 25)
            guidance = float(input("Guidance scale (default 7.5): ") or 7.5)
            width    = int(input("Width (default 512): ") or 512)
            height   = int(input("Height (default 512): ") or 512)
        except ValueError:
            steps, guidance, width, height = 25, 7.5, 512, 512

        print("\n🚀 Generating image via HF API...")
        image, info = generator.generate_image(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_steps=steps,
            guidance_scale=guidance,
            seed=-1,
            model_name=model_name
        )
        print(info)
        if image:
            image.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_generate()
    else:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True
        )