# Core model integration with Gemini API

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")


class Response:
    """Simple response wrapper for consistent interface."""
    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"Response(content={self.content[:50]}...)"


class GeminiModel:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize Gemini model."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name
        logger.info(f"GeminiModel initialized with model: {model_name}")

    def invoke(self, messages: List[Dict[str, str]]) -> Response:
        """Send messages to Gemini and return the response."""
        try:
            # Building the prompt by concatenating messages with role indicators
            prompt_parts = []
            for msg in messages:
                role = msg["role"].upper()
                prompt_parts.append(f"[{role}]: {msg['content']}")
            prompt = "\n\n".join(prompt_parts)

            logger.debug(f"Invoking Gemini with {len(messages)} messages")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response from Gemini API")
                return Response("Error: Empty response from model")

            logger.info("Gemini response generated successfully")
            return Response(content=response.text)

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return Response(f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error invoking Gemini model: {str(e)}", exc_info=True)
            return Response(f"Error: {str(e)}")


def get_model() -> GeminiModel:
    """Factory function to create a GeminiModel instance."""
    try:
        return GeminiModel()
    except ValueError as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
