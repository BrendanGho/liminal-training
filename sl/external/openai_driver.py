import asyncio
from typing import Literal
from openai.types import FileObject
from sl.llm.data_models import LLMResponse, Chat
from sl import config
from sl.llm.services import SampleCfg
from sl.utils import fn_utils
from loguru import logger
import openai


_client = None

def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(
            api_key=config.OPENAI_API_KEY or None,
        )
    return _client

@fn_utils.auto_retry_async([Exception], max_retry_attempts=10)
@fn_utils.max_concurrency_async(max_size=50)
async def sample(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    kwargs = sample_cfg.model_dump()
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]

    api_response = await get_client().chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], model=model_id, **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    total = len(input_chats)
    completed = 0
    logger.info(f"Sampling {total} responses from {model_id}...")

    async def _tracked_sample(chat: Chat, cfg: SampleCfg) -> LLMResponse:
        nonlocal completed
        result = await sample(model_id, chat, cfg)
        completed += 1
        if completed % 500 == 0 or completed == total:
            logger.info(f"  {completed}/{total} completed ({100*completed//total}%)")
        return result

    results = await asyncio.gather(
        *[_tracked_sample(c, s) for (c, s) in zip(input_chats, sample_cfgs)],
        return_exceptions=True,
    )
    failures = sum(1 for r in results if isinstance(r, Exception))
    if failures:
        logger.warning(f"{failures}/{total} samples failed and will be skipped")
    return [r for r in results if not isinstance(r, Exception)]


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)
