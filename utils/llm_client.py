import os
import yaml
from openai import OpenAI

config_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config",
    "config.yaml"
)

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    api_key=config["deepseek"]["api_key"],
    base_url=config["deepseek"]["base_url"]
)

def chat(prompt, model=None, temperature=0.7, max_tokens=500):
    resp = client.chat.completions.create(
        model=model or config["deepseek"]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content