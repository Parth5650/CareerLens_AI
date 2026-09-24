import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_text(prompt: str) -> str:
    """
    Calls a HuggingFace Inference API model using the provided prompt.
    Uses the OpenAI-compatible chat completions endpoint at router.huggingface.co/v1.
    """

    # 1. Get API token
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token or api_token == "your_huggingface_token_here":
        print("[WARNING] Using MOCK LLM response because no valid API token was provided.")
        # Return a mock JSON response for resume analysis so the app doesn't crash
        return """{
          "name": "Mock Candidate",
          "email": "mock@example.com",
          "skills": ["Python", "FastAPI", "React", "AI"],
          "education": ["B.S. Computer Science"],
          "experience": ["Software Engineer at Tech Corp"],
          "projects": ["CareerLens AI Clone"],
          "strengths": ["Fast learner", "Problem solver"],
          "weaknesses": ["Needs to configure API keys"],
          "suggestions": ["Add a real HUGGINGFACEHUB_API_TOKEN to the backend .env file!"]
        }"""

    # 2. Define API URL (OpenAI-compatible endpoint on HuggingFace)
    # model = "meta-llama/Llama-3.1-8B-Instruct"
    model = "openai/gpt-oss-120b"
    api_url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # 3. Build the chat-completions payload
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
    }

    # 4. Send request using requests library
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() 
        data = response.json()

        # 5. Extract and return the assistant's reply
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as errh:
        raise RuntimeError(f"HTTP Error: {errh} - Details: {response.text}")
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"Request Error: {err}")
    except (KeyError, IndexError) as err:
        raise RuntimeError(f"Unexpected response format: {data}")
