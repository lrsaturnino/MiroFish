"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Literal, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        role: Optional[Literal["builder", "swarm", "judge"]] = None,
    ):
        """
        构造LLM客户端，按优先级解析凭据来源。

        凭据解析优先级（每个字段独立）：显式kwarg > role分组 > 全局 LLM_*。
        当 ``role`` 提供时，未显式传入的字段经由 ``Config.llm_for(role)``
        解析；不提供 ``role`` 时退回原有的 ``Config.LLM_*`` 行为。

        Args:
            api_key: 显式 API key，优先级最高
            base_url: 显式 base URL，优先级最高
            model: 显式模型名，优先级最高
            reasoning_effort: 推理强度（仅GPT-5/o系列模型支持）
            role: ``"builder"`` / ``"swarm"`` / ``"judge"`` 之一；触发
                按角色分组的 ``BUILDER_LLM_*`` / ``SWARM_LLM_*`` /
                ``JUDGE_LLM_*`` 环境变量解析

        Example:
            >>> LLMClient(role="builder")  # 使用 BUILDER_LLM_* 环境变量
        """
        # Precedence: explicit kwarg > role group > global LLM_*
        if role is not None:
            role_key, role_url, role_model = Config.llm_for(role)
        else:
            role_key = role_url = role_model = None
        self.api_key = api_key or role_key or Config.LLM_API_KEY
        self.base_url = base_url or role_url or Config.LLM_BASE_URL
        self.model = model or role_model or Config.LLM_MODEL_NAME
        # Empty string is treated as "omit the param" so providers that don't
        # accept reasoning_effort (Qwen, DeepSeek, older OpenAI models) still work.
        effort = reasoning_effort if reasoning_effort is not None else Config.LLM_REASONING_EFFORT
        self.reasoning_effort = effort or None

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        # OpenAI GPT-5 and o-series reasoning models reject the legacy
        # `max_tokens` parameter and require `max_completion_tokens`. They
        # also reject any non-default `temperature`, accepting only the
        # model default (1). Other OpenAI-SDK-compatible providers (Qwen,
        # DeepSeek, etc.) still use both `max_tokens` and `temperature`,
        # so we gate these on the OpenAI reasoning family prefix.
        model_lc = (self.model or "").lower()
        is_reasoning_family = model_lc.startswith(("gpt-5", "o1", "o3", "o4"))
        if is_reasoning_family:
            # Reasoning tokens come out of the same `max_completion_tokens`
            # budget as output tokens. If `reasoning_effort` is high/xhigh
            # and the caller's budget is small (e.g. 4096), the model can
            # consume the entire budget on internal reasoning and return
            # empty content — which then fails downstream JSON parsing.
            # We enforce a floor sized by effort to guarantee output room.
            effort = (self.reasoning_effort or "").lower()
            reasoning_floor = {
                "xhigh": 32768,
                "high": 16384,
                "medium": 8192,
                "low": 4096,
                "minimal": 2048,
                "none": 2048,
            }.get(effort, 0)
            kwargs["max_completion_tokens"] = max(max_tokens, reasoning_floor)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        # reasoning_effort is only supported by GPT-5 and o-series reasoning
        # models. Forwarding it to Qwen/DeepSeek/etc. would error out, so we
        # gate on the same family check as max_completion_tokens.
        if self.reasoning_effort and is_reasoning_family:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

