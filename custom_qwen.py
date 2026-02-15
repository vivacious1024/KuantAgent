
from typing import Any, List, Optional, Dict, Union, Sequence, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

import dashscope
from dashscope import MultiModalConversation
import json

class CustomChatQwen(BaseChatModel):
    """
    Custom LangChain wrapper for Qwen models using DashScope SDK directly.
    Supports Qwen-Omni models via MultiModalConversation.
    Simulates tool calling via prompt engineering since standard API might not support it for this model.
    """
    model_name: str = "qwen3-omni-flash-2025-12-01"
    dashscope_api_key: Optional[str] = None
    temperature: float = 0.1
    max_retries: int = 3
    bound_tools: Optional[List[Any]] = None

    @property
    def _llm_type(self) -> str:
        return "custom_qwen"

    def bind_tools(self, tools: Sequence[Union[Dict[str, Any], Any, Callable, BaseTool]], **kwargs: Any) -> Runnable:
        """
        Bind tool definitions to the model.
        Since we are shimming tool support, we just store them to inject into the prompt later.
        """
        self.bound_tools = tools
        return self

    def _format_tools_prompt(self, tools):
        prompt = "\n\nYou have access to the following tools:\n"
        for tool in tools:
            name = getattr(tool, "name", str(tool))
            desc = getattr(tool, "description", "")
            args = getattr(tool, "args", {})
            prompt += f"- {name}: {desc}\n  Arguments: {args}\n"
        
        prompt += """\nTo use a tool, YOU MUST output a JSON object in this exact format (do not wrap in markdown):
{"tool": "tool_name", "parameters": {"arg_name": "value"}}

If you want to provide a final answer or analysis, just output the text normally without JSON.
"""
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        if self.dashscope_api_key:
            dashscope.api_key = self.dashscope_api_key
            
        # Format messages for DashScope
        dashscope_messages = []
        
        # Inject system prompt with tools if needed
        system_content = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content += msg.content + "\n"
        
        if self.bound_tools:
            system_content += self._format_tools_prompt(self.bound_tools)
            
        if system_content:
            dashscope_messages.append({"role": "system", "content": [{"text": system_content}]})

        # Process other messages
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue # Already handled
                
            role = "user"
            content_text = msg.content
            
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
                # If the AI message had tool calls, we should represent that in history
                # But for simplicity in this shim, we just use the text content if any
                if not content_text and msg.tool_calls:
                    # Provide a pseudo-text representation of what happened
                    content_text = f"Tool Called: {msg.tool_calls[0]['name']}"
            elif isinstance(msg, ToolMessage):
                role = "user" # DashScope Omni might not have 'tool' role, treat as user feedback
                content_text = f"Tool '{msg.name}' Output: {msg.content}"
            
            if content_text:
                if isinstance(content_text, str):
                    dashscope_messages.append({"role": role, "content": [{"text": content_text}]})
                elif isinstance(content_text, list):
                    # Handle multimodal content (list of dicts)
                    ds_content = []
                    for item in content_text:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                ds_content.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                url_info = item.get("image_url", {})
                                if isinstance(url_info, dict):
                                    ds_content.append({"image": url_info.get("url", "")})
                                else:
                                    ds_content.append({"image": url_info})
                    dashscope_messages.append({"role": role, "content": ds_content})

        try:
            # Call API
            response = MultiModalConversation.call(
                model=self.model_name,
                messages=dashscope_messages,
                result_format='message'
            )
            
            if response.status_code == 200:
                output_text = response.output.choices[0].message.content[0]['text']
                
                # Check for tool call in JSON
                tool_calls = []
                final_content = output_text
                
                try:
                    # Look for JSON-like structure
                    if "{" in output_text and "}" in output_text:
                        # Naive extraction
                        start = output_text.find("{")
                        end = output_text.rfind("}") + 1
                        potential_json = output_text[start:end]
                        
                        data = json.loads(potential_json)
                        if "tool" in data and "parameters" in data:
                            # It is a shimmed tool call
                            tool_calls.append({
                                "name": data["tool"],
                                "args": data["parameters"],
                                "id": f"call_{hash(output_text)}"
                            })
                            final_content = "" # No text content for tool call message
                except:
                    pass # Not a tool call, just text
                
                msg = AIMessage(content=final_content, tool_calls=tool_calls)
                return ChatResult(generations=[ChatGeneration(message=msg)])
            else:
                raise ValueError(f"DashScope API Error: {response.code} - {response.message}")
                
        except Exception as e:
            raise ValueError(f"Error calling DashScope: {str(e)}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
