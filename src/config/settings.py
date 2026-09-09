"""
配置管理模块

使用 Pydantic Settings 管理项目配置
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """项目配置"""

    # 阿里云通义千问 API Key
    DASHSCOPE_API_KEY: str = Field(
        env="DASHSCOPE_API_KEY",
        description="阿里云通义千问 API Key"
    )

    # 模型配置
    QWEN_TURBO_MODEL: str = Field(
        default="qwen-turbo",
        env="QWEN_TURBO_MODEL",
        description="Turbo 模型名称"
    )

    QWEN_MAX_MODEL: str = Field(
        default="qwen-max-longcontext",
        env="QWEN_MAX_MODEL",
        description="Max 模型名称"
    )

    # 路由配置
    COMPLEXITY_TOKEN_THRESHOLD: int = Field(
        default=500,
        env="COMPLEXITY_TOKEN_THRESHOLD",
        description="Token 阈值，超过此长度视为复杂任务"
    )

    MAX_RETRIES: int = Field(
        default=3,
        env="MAX_RETRIES",
        description="最大重试次数"
    )

    # 环境配置
    ENV: str = Field(
        default="dev",
        env="ENV",
        description="环境标识"
    )

    # 日志配置
    LOG_LEVEL: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="日志级别"
    )

    # 是否启用 LangGraph
    ENABLE_LANGGRAPH: bool = Field(
        default=True,
        env="ENABLE_LANGGRAPH",
        description="是否启用 LangGraph"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置实例
settings = Settings()


# 便捷函数
def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def validate_settings() -> bool:
    """验证配置是否有效"""
    if not settings.DASHSCOPE_API_KEY:
        print("错误: DASHSCOPE_API_KEY 未配置")
        return False

    return True


if __name__ == "__main__":
    # 测试配置
    settings = get_settings()
    print("配置加载成功:")
    print(f"  Turbo 模型: {settings.QWEN_TURBO_MODEL}")
    print(f"  Max 模型: {settings.QWEN_MAX_MODEL}")
    print(f"  Token 阈值: {settings.COMPLEXITY_TOKEN_THRESHOLD}")
    print(f"  最大重试次数: {settings.MAX_RETRIES}")
    print(f"  环境: {settings.ENV}")
    print(f"  LangGraph 启用: {settings.ENABLE_LANGGRAPH}")

    # 验证配置
    if validate_settings():
        print("\n✅ 配置验证通过")
    else:
        print("\n❌ 配置验证失败")
