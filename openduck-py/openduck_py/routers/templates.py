from typing import Dict, List, Any
import os

from litellm import acompletion, ModelResponse
import jinja2

from openduck_py.settings import CHAT_MODEL

# to not break existing env files
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2023-05-15"


# TODO (Matthew): Move these to utils, or just call the chat completions API without wrapper functions


async def chat_continuation(
    messages: List[Dict[str, str]], model: str = CHAT_MODEL
) -> ModelResponse:
    response = await acompletion(model=model, messages=messages, temperature=0.3)
    return response


async def generate(
    template: str,
    variables: Dict[str, Any],
    model: str = CHAT_MODEL,
    role="user",
) -> ModelResponse:
    jinja_template = jinja2.Template(template)

    prompt = [{"content": jinja_template.render(variables), "role": role}]

    response = await acompletion(model=model, messages=prompt, temperature=0.3)
    return response
