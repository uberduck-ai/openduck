from typing import Dict, List, Any, Literal
import os

from pydantic import BaseModel
from openai import AsyncAzureOpenAI
import jinja2

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
)


ModelLiteral = Literal["gpt-35-turbo-deployment", "gpt-4-deployment"]
DEFAULT_MODEL: ModelLiteral = "gpt-35-turbo-deployment"


class GenerationResponse(BaseModel):
    """
    OpenAI Chat Completions response format:
    https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    """

    choices: List
    created: int
    id: str
    model: str
    object: str
    usage: Any


# TODO (Matthew): Move these to utils, or just call the chat completions API without wrapper functions

async def open_ai_chat_continuation(
    messages: List[Dict[str, str]], model: ModelLiteral
) -> GenerationResponse:
    response = await client.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=messages, temperature=0.3
    )
    return response


async def generate(
    template: str,
    variables: Dict[str, Any],
    model: ModelLiteral,
    role="user",
) -> GenerationResponse:
    jinja_template = jinja2.Template(template)

    prompt = [{"content": jinja_template.render(variables), "role": role}]

    response = await client.chat.completions.create(
        model=model or DEFAULT_MODEL, messages=prompt, temperature=0.3
    )
    return response
