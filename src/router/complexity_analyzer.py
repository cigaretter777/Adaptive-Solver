"""
任务复杂度分析器

负责分析用户输入的任务复杂度，为动态路由提供决策依据。
分析维度包括：
1. Token 长度
2. 关键词意图识别
3. 任务类型判断
4. 多步推理检测
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    SIMPLE_QA = "simple_qa"           # 简单问答
    CALCULATION = "calculation"       # 计算
    REASONING = "reasoning"           # 推理
    PROOF = "proof"                   # 证明
    ANALYSIS = "analysis"             # 分析
    CODE_GEN = "code_generation"      # 代码生成
    MULTI_STEP = "multi_step"         # 多步骤任务
    UNKNOWN = "unknown"               # 未知


class ComplexityLevel(Enum):
    """复杂度等级"""
    SIMPLE = "simple"       # 简单
    MEDIUM = "medium"       # 中等
    COMPLEX = "complex"     # 复杂


@dataclass
class ComplexityAnalysis:
    """复杂度分析结果"""
    level: ComplexityLevel
    task_type: TaskType
    token_count: int
    confidence: float
    reasons: List[str]
    features: Dict[str, any]

    def is_complex(self) -> bool:
        """判断是否为复杂任务"""
        return self.level in [ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]


class ComplexityAnalyzer:
    """
    任务复杂度分析器

    使用规则 + 特征提取的方式分析任务复杂度
    """

    # 复杂任务关键词
    COMPLEX_KEYWORDS = [
        "证明", "推导", "分析", "评估", "比较",
        "设计", "优化", "实现", "调试", "定位",
        "explain", "prove", "derive", "analyze", "evaluate",
        "design", "optimize", "implement", "debug", "locate"
    ]

    # 多步骤关键词
    MULTI_STEP_KEYWORDS = [
        "首先", "然后", "接着", "最后", "步骤",
        "first", "then", "next", "finally", "step",
        "依次", "按照", "分步", "流程"
    ]

    # 简单任务模式
    SIMPLE_PATTERNS = [
        r"^计算\s*\d+[\+\-\*\/]\s*\d+",  # 简单计算
        r"^\d+\s*[\+\-\*\/]\s*\d+\s*[=？?]",  # 简单等式
        r"^(是|否|对|错)\?",  # 简单判断
        r"^(what|who|when|where|why|how many)\s+",  # 简单问题
        r"^[\d一二三四五六七八九十]+[\s、，.，、]",  # 简单列举
    ]

    # 推理任务关键词
    REASONING_KEYWORDS = [
        "为什么", "如何", "怎样", "推理", "逻辑",
        "why", "how", "reasoning", "logic", "deduce"
    ]

    # 数学证明关键词
    PROOF_KEYWORDS = [
        "证明", "证", "定理", "命题", "假设",
        "prove", "proof", "theorem", "proposition", "hypothesis",
        "矛盾", "反证", "归纳"
    ]

    # 代码生成关键词
    CODE_KEYWORDS = [
        "代码", "函数", "类", "算法", "实现",
        "code", "function", "class", "algorithm", "implement",
        "编程", "写一个", "Python", "Java", "JavaScript"
    ]

    def __init__(self, token_threshold: int = 500):
        """
        初始化分析器

        Args:
            token_threshold: Token 阈值，超过视为复杂任务
        """
        self.token_threshold = token_threshold

    def analyze(self, query: str, history: Optional[List[Dict]] = None) -> ComplexityAnalysis:
        """
        分析任务复杂度

        Args:
            query: 用户输入的查询
            history: 对话历史，用于判断多轮对话

        Returns:
            ComplexityAnalysis: 复杂度分析结果
        """
        reasons = []
        features = {}
        confidence = 1.0

        # 1. Token 长度分析
        token_count = self._estimate_tokens(query)
        features["token_count"] = token_count

        if token_count > self.token_threshold * 2:
            reasons.append(f"Token 长度 {token_count} 超过 2 倍阈值")

        # 2. 任务类型识别
        task_type = self._classify_task_type(query)
        features["task_type"] = task_type.value

        # 3. 复杂度评分
        complexity_score = self._calculate_complexity_score(query, history, task_type)
        features["complexity_score"] = complexity_score

        # 4. 确定复杂度等级
        if complexity_score >= 0.7:
            level = ComplexityLevel.COMPLEX
        elif complexity_score >= 0.4:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.SIMPLE

        # 5. 收集判断依据
        reasons.extend(self._get_reasons(query, task_type, complexity_score))

        # 6. 计算置信度
        confidence = self._calculate_confidence(complexity_score, token_count)
        features["confidence"] = confidence

        return ComplexityAnalysis(
            level=level,
            task_type=task_type,
            token_count=token_count,
            confidence=confidence,
            reasons=reasons,
            features=features
        )

    def _estimate_tokens(self, text: str) -> int:
        """
        估算 Token 数量

        简单估算：中文字符按 1.5 token 计，英文单词按 1 token 计
        """
        if not text:
            return 0

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - english_words

        # 简单估算公式
        return int(chinese_chars * 1.5 + english_words + other_chars * 0.5)

    def _classify_task_type(self, query: str) -> TaskType:
        """分类任务类型"""
        query_lower = query.lower()

        # 检查证明类任务
        if any(kw in query for kw in self.PROOF_KEYWORDS):
            return TaskType.PROOF

        # 检查代码生成任务
        if any(kw in query for kw in self.CODE_KEYWORDS):
            return TaskType.CODE_GEN

        # 检查多步骤任务
        if any(kw in query for kw in self.MULTI_STEP_KEYWORDS):
            return TaskType.MULTI_STEP

        # 检查推理任务
        if any(kw in query for kw in self.REASONING_KEYWORDS):
            return TaskType.REASONING

        # 检查分析任务
        if any(kw in query for kw in self.COMPLEX_KEYWORDS):
            return TaskType.ANALYSIS

        # 检查计算任务
        if re.search(r'[\d]+\s*[\+\-\*\/\^]\s*[\d]+', query):
            return TaskType.CALCULATION

        # 默认为简单问答
        return TaskType.SIMPLE_QA

    def _calculate_complexity_score(self, query: str, history: Optional[List[Dict]],
                                    task_type: TaskType) -> float:
        """
        计算复杂度分数 (0-1)

        考虑因素：
        - Token 长度
        - 任务类型
        - 历史对话轮数
        - 关键词复杂度
        """
        score = 0.0

        # 1. Token 长度得分 (0-0.3)
        token_count = self._estimate_tokens(query)
        if token_count > self.token_threshold * 2:
            score += 0.3
        elif token_count > self.token_threshold:
            score += 0.2
        elif token_count > self.token_threshold / 2:
            score += 0.1

        # 2. 任务类型得分 (0-0.4)
        type_scores = {
            TaskType.PROOF: 0.4,
            TaskType.MULTI_STEP: 0.35,
            TaskType.CODE_GEN: 0.35,
            TaskType.ANALYSIS: 0.3,
            TaskType.REASONING: 0.25,
            TaskType.CALCULATION: 0.1,
            TaskType.SIMPLE_QA: 0.0,
            TaskType.UNKNOWN: 0.15
        }
        score += type_scores.get(task_type, 0.15)

        # 3. 历史对话轮数得分 (0-0.15)
        if history:
            round_count = len(history)
            if round_count >= 5:
                score += 0.15
            elif round_count >= 3:
                score += 0.1
            elif round_count >= 1:
                score += 0.05

        # 4. 关键词复杂度 (0-0.15)
        query_lower = query.lower()
        complex_kw_count = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in query_lower)
        score += min(complex_kw_count * 0.05, 0.15)

        return min(score, 1.0)

    def _get_reasons(self, query: str, task_type: TaskType, score: float) -> List[str]:
        """获取复杂度判断的理由"""
        reasons = []

        if score >= 0.7:
            reasons.append("综合复杂度评分 >= 0.7，判定为复杂任务")

        if task_type in [TaskType.PROOF, TaskType.MULTI_STEP]:
            reasons.append(f"任务类型为 {task_type.value}，需要多步推理")

        if any(kw in query.lower() for kw in self.COMPLEX_KEYWORDS):
            reasons.append("包含复杂任务关键词")

        token_count = self._estimate_tokens(query)
        if token_count > self.token_threshold:
            reasons.append(f"输入长度 {token_count} tokens 超过阈值 {self.token_threshold}")

        return reasons

    def _calculate_confidence(self, complexity_score: float, token_count: int) -> float:
        """
        计算判断置信度

        边界情况的置信度较低
        """
        # 复杂度分数接近边界 (0.4 或 0.7) 时，置信度降低
        if abs(complexity_score - 0.4) < 0.1 or abs(complexity_score - 0.7) < 0.1:
            return 0.7

        # Token 长度很小时，判断更可靠
        if token_count < 50:
            return 0.95

        # Token 长度很大时，判断也更可靠
        if token_count > self.token_threshold * 2:
            return 0.95

        return 0.85


# 便捷函数
def analyze_complexity(query: str, history: Optional[List[Dict]] = None,
                      token_threshold: int = 500) -> ComplexityAnalysis:
    """
    便捷函数：快速分析任务复杂度

    Args:
        query: 用户输入
        history: 对话历史
        token_threshold: Token 阈值

    Returns:
        ComplexityAnalysis: 分析结果
    """
    analyzer = ComplexityAnalyzer(token_threshold=token_threshold)
    return analyzer.analyze(query, history)


if __name__ == "__main__":
    # 测试示例
    test_queries = [
        "计算 123 * 456 的结果",
        "证明：任意两个连续整数的乘积是偶数",
        "分析这个算法的时间复杂度，并给出优化建议",
        "Python 中如何实现一个二叉树的遍历？",
        "设计一个用户认证系统，包含注册、登录和密码找回功能"
    ]

    analyzer = ComplexityAnalyzer(token_threshold=500)

    for query in test_queries:
        result = analyzer.analyze(query)
        print(f"\n查询: {query}")
        print(f"  复杂度: {result.level.value}")
        print(f"  任务类型: {result.task_type.value}")
        print(f"  Token 数: {result.token_count}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  理由: {', '.join(result.reasons)}")
