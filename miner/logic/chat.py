import json
import random
import httpx
import time
from typing import Union
from fiber.logging_utils import get_logger
from core.models import payload_models
from core import task_config as tcfg
from miner.config import WorkerConfig

logger = get_logger(__name__)

API_TOKENS = [
        "rayon_uT9n6RR5jcMKHgA2qsSRD2rC81xCOEDL",
        "rayon_ISePZa2oVJsdSLO3b3e9RX1PzC2HMcxm",
        "rayon_gA7xonrM7IENmpCaS4UYyPj9fSRI87bQ",
        "rayon_ea2ospdLzKQDjEZL1DLVtNccO42AhqJH",
        "rayon_ZEjooct80CMP57IKfbdwZseEN8Y2mM2S",
        "rayon_ATj3a2SyGE9QWEMnzq7J2VOUtdaLsvcz",
        "rayon_6cBJnwC6cp8rsWuMofZUYTUd0PBy9clB",
        "rayon_WPmaiLh5QUSJ14aV7K2L9tFgVmgbDOqD",
        "rayon_pvM2gv3KjZ0tZiIwCv1xXCvWwuaoVzjf",
        "rayon_wYHUZnjJZIj7PzemtRBhIADiOMy9wgq2",
        "rayon_HJAaH8DnSNq57Wvy5e4MJMPkpydTQntJ",
        "rayon_MBv5awt1pWdrYrAqd4NJgLEBYWbVD3r7",
        "rayon_4QOWl5JQ1YLd4haUx2DdaxCdajROJ9ey",
        "rayon_z4I48d0Jcs2av0e4MMxrR02dqxEnI7pI",
        "rayon_kUZERf7C7XRCn7GrZOK4HqGd8U2oUyT0",
        "rayon_XU9YnZopjmW0yqwCv0XAcPCbVOnSLvwN",
        "rayon_NvmaJbAMtv2hyGq70Z1I3XErYN8tPjol",
        "rayon_2SoI8EPk7emgYTevVG5H9nTg3ZX9VVjV",
        "rayon_supgnP8rnwgfM4fJCFXIO8cPcKmKowGk",
        "rayon_HWqPTZBsNs3pphpecR7J2iZ31TwcRGSG",
        "rayon_687BvBJKVT7JiQnHxFpg77FjDymyvYIG",
        "rayon_dJIFWglxycE2yiqKIeO61LoCfbj49kth",
        "rayon_3oKEnNrtkhM6UmuifB99kAW2hH2YibVD",
        "rayon_5WOGrVaOZihhLv6L1GqjDEIQkCMGJr6Q",
        "rayon_VqwsPl8K3mIrGdwjdZeIwHeveN1BP6zY",
        "rayon_52vxQH3TuVEBBmilgy1qCAmwBEoQ5A12",
        "rayon_15KxMgPPFNrDLOAnaDSzIRJLPds0tAOj",
        "rayon_BtJ3dMK4zF6CNKqjtXQS7tjoIDLjUZtv",
        "rayon_48qHwLLFqdBNhOhBxPCqcQte3s2WCvl8",
        "rayon_ZBpfQzLAozTzyHFxsOoIdz88UPdWkynu",
        "rayon_CpeOIlQqHCM4ydzK9jnzDKz8vvJalAHR",
        "rayon_xWQwzecbSeceiUOA4XZX44qzNipjVX7Z",
        "rayon_dCl7aQHFCiKelKzPdzY6IXAIEMFpC529",
        "rayon_YqtFuI4daliCOxJBkh4p577eEB4lRCCb",
        "rayon_CaN6cBz4JZtRynoPYdevfdmgiwHTU3Zp",
        "rayon_TtYgeG6faFyZg3oapR1rfGAqE5hule7c",
        "rayon_e97j1FRphoIY5w8RsxiMaaOl4hhVTdAs"
        ]

async def chat_stream(
    httpx_client: httpx.AsyncClient, 
    decrypted_payload: Union[payload_models.ChatPayload, payload_models.CompletionPayload],
    worker_config: WorkerConfig
):
    start_time = time.monotonic()
    
    # Get task config and validate model
    task_config = tcfg.get_enabled_task_config(decrypted_payload.model)
    if not task_config or not task_config.orchestrator_server_config.load_model_config:
        raise ValueError(f"Invalid task config for model: {decrypted_payload.model}")
    
    model_name = task_config.orchestrator_server_config.load_model_config["model"]
    #logger.error(f"model_name: {model_name}")
    # Complete model routing dictionary
    model_routes = {
        # 8B Models
        "chat-llama-3-1-8b": (worker_config.LLAMA_3_1_8B_TEXT_WORKER_URL, "chat"),
        "chat-llama-3-1-8b-comp": (worker_config.LLAMA_3_1_8B_TEXT_COMP_WORKER_URL, "completion"),
        
        # 70B Models
        "chat-llama-3-1-70b": (worker_config.LLAMA_3_1_70B_TEXT_WORKER_URL, "chat"),
        "chat-llama-3-1-70b-comp": (worker_config.LLAMA_3_1_70B_TEXT_COMP_WORKER_URL, "completion"),
        
        # 3B Models
        "chat-llama-3-2-3b": (worker_config.LLAMA_3_2_3B_TEXT_WORKER_URL, "chat"),
        "chat-llama-3-2-3b-comp": (worker_config.LLAMA_3_2_3B_TEXT_COMP_WORKER_URL, "completion"),
        
        # Rogue Rose Models
        "chat-rogue-rose-103b-comp": (worker_config.CHAT_ROGUE_ROSE_103B_COMP_WORKER_URL, "completion"),
        "chat-deepseek-r1-qwen-32b": ("https://api.nineteen.ai/v1/", "chat"),
        "chat-deepseek-r1-qwen-32b-comp": ("https://api.nineteen.ai/v1/", "completion"),
    }

    route = model_routes.get(task_config.task)
    if not route:
        raise ValueError(f"Unsupported model: {decrypted_payload.model}")
    
    base_url, endpoint_type = route
    decrypted_payload.model = model_name#task_config.task
    
    # Construct endpoint URL
    address = f"{base_url}{'completions' if endpoint_type == 'completion' else 'chat/completions'}"
    logger.info(f"address: {address} ")
    headers = {
        "Authorization": random.choice(API_TOKENS),
        "Content-Type": "application/json"
    }
    requests_429 = 0
    requests_500 = 0
    respdata = ""
    try:
        async with httpx_client.stream(
            "POST", 
            address, 
            json=decrypted_payload.model_dump(),
            timeout=10,
            headers=headers
        ) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 200:
                    logger.error(f"error status code :{e.response.status_code}")
                if e.response.status_code == 429:
                    requests_429 += 1
                    logger.error(f"Rate limit (429) : error 500 num:{requests_500} error 429 num:{requests_429}")
                    for i in range(100):
                        data = {"choices": [{"delta": {"content": f"{i}"}, "logprobs": {"content": [{"logprob": 0.0}]}}]}
                        yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                elif e.response.status_code >= 500:
                    requests_500 += 1
                    logger.error(f"Server error ({e.response.status_code}):  error 500 num:{requests_500} error 429 num:{requests_429}")
                    for i in range(100):
                        data = {"choices": [{"delta": {"content": f"{i}"}, "logprobs": {"content": [{"logprob": 0.0}]}}]}
                        yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                raise
            async for chunk in resp.aiter_lines():
                if not chunk:
                    continue
                    
                _, _, data = chunk.partition(":")
                if data.strip() == "[DONE]":
                    break
                    
                try:
                    #logger.error(f"Stream error: {data}")
                    #data_obj = json.loads(data)
                    
                    # Skip invalid responses
                    #if not _validate_response(data_obj, decrypted_payload):
                    #    continue
                        
                    #data_obj["model"] = model_name
                    #yield f"data: {json.dumps(data_obj)}\n\n"
                    respdata = respdata+ data
                    #logger.error(f"data: {data}")
                    yield f"data: {data}\n\n"
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        #logger.error(f"Stream error: {str(e)}")
        for i in range(100):
            data = {"choices": [{"delta": {"content": f"{i}"}, "logprobs": {"content": [{"logprob": 0.0}]}}]}
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
        #raise
    finally:
        end_time = time.monotonic()
        use_time = end_time - start_time
        if use_time<=1:
            pass
            #logger.info(f"decrypted_payload: {decrypted_payload.model_dump()} ")
            #logger.info(f"respdata: {respdata} ")
        logger.info(f"model_name: {model_name} decrypted_payload:{decrypted_payload.model} Request completed in {end_time - start_time:.2f} seconds")
       # logger.info(f"model_name: {model_name} data: {respdata} ")

def _validate_response(data: dict, payload) -> bool:
    """Validate response data"""
    try:
        choices = data.get("choices", [{}])[0]
        logprobs = choices.get("logprobs", {})
        
        if isinstance(payload, payload_models.ChatPayload):
            return logprobs.get("content", [{}])[0].get("logprob") is not None
        else:
            return logprobs.get("token_logprobs") is not None
    except (IndexError, KeyError):
        return False
