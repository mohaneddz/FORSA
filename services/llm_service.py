import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config import config
from utils.logging_utils import log_execution_time

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_url = config.DEEPSEEK_API_URL
        self.token = config.DEEPSEEK_TOKEN
        self.model = config.MODEL_NAME

    @log_execution_time
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, json_mode: bool = False, max_retries: int = 3) -> str:
        """
        Send a chat completion request to DeepSeek.
        If json_mode=True, enforces JSON output and validates the response.
        Retries on invalid JSON up to max_retries times.
        """
        if not self.token:
            raise ValueError("DEEPSEEK_TOKEN is not set.")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to DeepSeek API. Model: {self.model}, Attempt: {attempt + 1}/{max_retries}")
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Log usage if available
                if "usage" in data:
                    usage = data["usage"]
                    logger.info(f"Token Usage - Prompt: {usage.get('prompt_tokens')}, Completion: {usage.get('completion_tokens')}, Total: {usage.get('total_tokens')}")
                
                content = data["choices"][0]["message"]["content"]
                
                # If JSON mode is enabled, validate the response
                if json_mode:
                    content = self._validate_and_clean_json(content)
                
                return content
                
            except requests.exceptions.RequestException as e:
                logger.error(f"DeepSeek API Error (attempt {attempt + 1}/{max_retries}): {e}")
                if e.response:
                    logger.error(f"Response Content: {e.response.text}")
                
                if attempt == max_retries - 1:
                    raise
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid JSON response (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get valid JSON after {max_retries} attempts")
                    raise ValueError(f"LLM returned invalid JSON after {max_retries} attempts: {e}")
        
        raise RuntimeError("Unexpected error in chat_completion retry logic")
    
    def _validate_and_clean_json(self, content: str) -> str:
        """
        Validate and clean JSON response from LLM.
        Handles cases where LLM wraps JSON in markdown code blocks.
        """
        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]  # Remove ```
        
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        
        content = content.strip()
        
        # Validate JSON
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            logger.error(f"JSON validation failed: {e}")
            logger.error(f"Content: {content[:500]}")  # Log first 500 chars
            raise
    
    def warmup(self) -> bool:
        """
        Send a warmup request to DeepSeek to initialize the service.
        This helps reduce latency on the first real request.
        
        Returns:
            True if warmup successful, False otherwise
        """
        try:
            logger.info("🔥 Warming up DeepSeek service...")
            
            warmup_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            response = self.chat_completion(
                messages=warmup_messages,
                temperature=0.1,
                json_mode=False,
                max_retries=2
            )
            
            logger.info(f"✓ DeepSeek warmup successful (response: {len(response)} chars)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠ DeepSeek warmup failed (non-critical): {e}")
            return False

    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        json_mode: bool = False
    ) -> str:
        """
        Async version of chat_completion for parallel processing.
        Makes non-blocking HTTP requests to the LLM API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            json_mode: Enable JSON response mode
            
        Returns:
            LLM response text
        """
        import aiohttp
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"LLM API error (attempt {attempt + 1}/{max_retries}): {response.status} - {error_text}")
                            
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                logger.info(f"Retrying in {delay}s...")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                raise Exception(f"LLM API error after {max_retries} attempts: {response.status}")
                        
                        data = await response.json()
                        
                        # Extract response
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            
                            # Log token usage if available
                            if "usage" in data:
                                usage = data["usage"]
                                logger.debug(f"LLM tokens - input: {usage.get('prompt_tokens')}, "
                                           f"output: {usage.get('completion_tokens')}, "
                                           f"total: {usage.get('total_tokens')}")
                            
                            return content
                        else:
                            raise Exception("Invalid LLM response format")
                        
            except asyncio.TimeoutError:
                logger.error(f"LLM API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise Exception(f"LLM API timeout after {max_retries} attempts")
                    
            except Exception as e:
                logger.error(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
        
        raise Exception("Unexpected error in LLM async completion")

llm_service = LLMService()
