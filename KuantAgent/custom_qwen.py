import os
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field


def _read_persistent_windows_env(var_name: str) -> str:
    try:
        import winreg  # type: ignore
    except Exception:
        return ""

    registry_paths = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ]
    for root, subkey in registry_paths:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, var_name)
                if value:
                    return str(value)
        except Exception:
            continue
    return ""


class CustomChatQwen(BaseChatModel):
    """Compatibility wrapper that routes Qwen-provider calls through SiliconFlow."""

    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    dashscope_api_key: Optional[str] = None
    siliconflow_api_key: Optional[str] = None
    mimo_api_key: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 2
    base_url: str = "https://api.siliconflow.cn/v1"
    _delegate: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        api_key = (
            self.siliconflow_api_key
            or self.dashscope_api_key
            or os.environ.get("SILICONFLOW_API_KEY", "")
            or _read_persistent_windows_env("SILICONFLOW_API_KEY")
            or self.mimo_api_key
            or os.environ.get("MIMO_API_KEY", "")
            or _read_persistent_windows_env("MIMO_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "SiliconFlow API key not found. Please set SILICONFLOW_API_KEY."
            )

        self._delegate = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    @property
    def _llm_type(self) -> str:
        return "custom_qwen_siliconflow"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._delegate.invoke(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])
