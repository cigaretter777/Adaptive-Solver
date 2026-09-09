"""
LLM 模型提供商封装

封装阿里云通义千问模型，提供统一的调用接口
"""
# 处理直接运行时的导入问题
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    # 切换到绝对导入
    from src.config.settings import get_settings

else:
    from ..config.settings import get_settings

from typing import Dict, Any, List, Optional, Union
class QwenProvider:
    """阿里云通义千问模型提供商"""

    def __init__(self, api_key: str = None, model: str = "qwen-turbo"):
        """
        初始化 Qwen 提供商

        Args:
            api_key: API Key
            model: 模型名称
        """
        self.settings = get_settings()
        self.model = model
        self.api_key = api_key or self.settings.DASHSCOPE_API_KEY

        # 简化版实现，实际使用时需要安装 dashscope SDK
        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        if not self.api_key:
            raise ValueError("Qwen API Key 未配置")

    def chat(self,
             messages: List[Dict[str, Any]],
             temperature: float = 0.7,
             max_tokens: int = 1500,
             **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 Token 数
            **kwargs: 其他参数

        Returns:
            Dict: 响应结果
        """
        # 注意：这里是示例实现，实际使用时需要安装 dashscope SDK
        # from dashscope import get_completion

        response = {
            "content": "这是一个模拟响应，实际使用时需要连接真实的 API",
            "model": self.model,
            "usage": {
                "prompt_tokens": len(str(messages)),
                "completion_tokens": max_tokens,
                "total_tokens": len(str(messages)) + max_tokens
            }
        }

        return response

    def invoke(self, messages: Union[str, List[Dict]], **kwargs) -> str:
        """
        调用模型（兼容 LangChain 接口）

        Args:
            messages: 消息
            **kwargs: 其他参数

        Returns:
            str: 响应文本
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = self.chat(messages, **kwargs)
        return response["content"]


class QwenTurboProvider(QwenProvider):
    """Qwen Turbo 提供商"""

    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key,
            model=get_settings().QWEN_TURBO_MODEL
        )


class QwenMaxProvider(QwenProvider):
    """Qwen Max 提供商"""

    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key,
            model=get_settings().QWEN_MAX_MODEL
        )


# 工厂函数
def get_qwen_provider(provider_type: str = "turbo", api_key: str = None) -> QwenProvider:
    """
    获取 Qwen 提供商实例

    Args:
        provider_type: 提供商类型 ("turbo" 或 "max")
        api_key: API Key

    Returns:
        QwenProvider: 提供商实例
    """
    if provider_type.lower() == "turbo":
        return QwenTurboProvider(api_key)
    elif provider_type.lower() == "max":
        return QwenMaxProvider(api_key)
    else:
        raise ValueError(f"不支持的提供商类型: {provider_type}")


# 便捷函数
def create_llm_client(provider_type: str = "turbo", **kwargs):
    """
    创建 LLM 客户端

    Args:
        provider_type: 提供商类型
        **kwargs: 其他参数

    Returns:
        QwenProvider: 客户端实例
    """
    return get_qwen_provider(provider_type, **kwargs)


if __name__ == "__main__":
    # 测试提供商
    print("测试 Qwen 提供商...")

    # 创建 Turbo 提供商
    turbo = create_llm_client("turbo")
    print(f"Turbo 模型: {turbo.model}")

    # 创建 Max 提供商
    max_provider = create_llm_client("max")
    print(f"Max 模型: {max_provider.model}")

    # 测试调用
    messages = [{"role": "user", "content": "你好，请介绍一下自己"}]
    try:
        response = turbo.chat(messages)
        print(f"响应: {response['content']}")
    except Exception as e:
        print(f"错误: {e}")
