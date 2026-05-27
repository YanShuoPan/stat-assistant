import openai
import os
import base64
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

prompts = [
    {
        "filename": "image1_knowledge_flow.png",
        "prompt": (
            "A clean, modern infographic-style illustration showing a knowledge pipeline for statistics research. "
            "On the left side, a researcher uploads academic papers and statistical method documents into a glowing digital platform. "
            "In the center, the platform processes and organizes the knowledge with neural network and AI visual elements. "
            "On the right side, end users ask questions and receive precise, knowledge-based answers on their screens. "
            "Use a professional blue and white color scheme with subtle data visualization elements like graphs and formulas in the background. "
            "Modern, flat design style suitable for an academic poster. No text or words in the image."
        ),
    },
    {
        "filename": "image2_ai_conversation.png",
        "prompt": (
            "A modern illustration of an AI-powered research assistant platform for statistics. "
            "Show a sleek chat interface on a large screen where a user is conversing with an AI assistant. "
            "The AI response area shows statistical formulas, charts, and referenced research paper icons. "
            "Around the screen, floating holographic elements represent statistical methods: regression lines, distribution curves, Bayesian networks, and hypothesis testing diagrams. "
            "Professional, clean design with a blue and teal color palette. "
            "Academic and technological atmosphere, suitable for a research project poster. No text or words in the image."
        ),
    },
]

os.makedirs("generated_images", exist_ok=True)

for item in prompts:
    print(f"Generating: {item['filename']}...")
    result = client.images.generate(
        model="gpt-image-1",
        prompt=item["prompt"],
        size="1536x1024",
        quality="high",
        n=1,
    )
    image_b64 = result.data[0].b64_json
    path = os.path.join("generated_images", item["filename"])
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"  Saved to {path}")

print("Done!")
