"""
状态机编排模块测试

测试工作流和各个节点的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph import (
    AgentState,
    create_initial_state,
    Planner,
    ToolCaller,
    Critic,
    Executor,
    Workflow,
    planner_node,
    tool_caller_node,
    critic_node,
    executor_node,
    run_workflow
)


def test_state_management():
    """测试状态管理"""
    print("1. 测试状态管理...")

    state = create_initial_state("测试查询")
    assert state["status"] == "planning"
    assert state["query"] == "测试查询"
    assert state["error_count"] == 0
    print("  [OK] 初始状态创建")

    # 测试消息更新
    from src.graph import update_message
    state = update_message(state, "assistant", "回复")
    assert len(state["messages"]) == 2
    print("  [OK] 消息更新")

    # 测试错误处理
    from src.graph import increment_error, reset_errors
    state = increment_error(state, "测试错误")
    assert state["error_count"] == 1
    state = reset_errors(state)
    assert state["error_count"] == 0
    print("  [OK] 错误处理")


def test_planner_node():
    """测试规划节点"""
    print("\n2. 测试规划节点...")

    state = create_initial_state("证明：素数有无穷多个")
    state = planner_node(state)

    assert state["plan"] is not None
    assert len(state["subtasks"]) > 0
    assert state["status"] == "executing"
    print("  [OK] 规划节点生成计划")


def test_tool_caller_node():
    """测试工具调用节点"""
    print("\n3. 测试工具调用节点...")

    state = create_initial_state("计算 1+1")
    state["subtasks"] = ["执行计算"]
    state = tool_caller_node(state)

    assert len(state["tool_calls"]) > 0
    print("  [OK] 工具调用节点执行")


def test_critic_node():
    """测试反思节点"""
    print("\n4. 测试反思节点...")

    # 测试正常情况
    state = create_initial_state("测试查询")
    state["subtasks"] = ["任务1"]
    state["tool_results"] = [{"task": "任务1", "result": "成功完成"}]
    state["current_subtask_index"] = 1  # 所有任务完成
    state = critic_node(state)

    assert state["criticism"] is not None
    print("  [OK] 反思节点执行")


def test_executor_node():
    """测试执行节点"""
    print("\n5. 测试执行节点...")

    state = create_initial_state("测试查询")
    state["plan"] = "测试计划"
    state["subtasks"] = ["任务1"]
    state["tool_results"] = [{"task": "任务1", "result": "测试结果"}]
    state["current_subtask_index"] = 1
    state = executor_node(state)

    assert state["answer"] is not None
    assert state["status"] == "completed"
    print("  [OK] 执行节点生成答案")


def test_workflow():
    """测试完整工作流"""
    print("\n6. 测试完整工作流...")

    workflow = Workflow()

    # 测试简单计算
    result = workflow.run("计算斐波那契数列前5项的和", verbose=False)

    assert result["answer"] is not None
    assert result["status"] in ["completed", "failed"]
    assert result["step_count"] > 0
    print(f"  [OK] 工作流执行完成 (状态: {result['status']}, 步骤: {result['step_count']})")


def test_convenience_function():
    """测试便捷函数"""
    print("\n7. 测试便捷函数...")

    result = run_workflow("测试查询", verbose=False)
    assert result is not None
    assert "answer" in result
    print("  [OK] 便捷函数执行")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("状态机编排模块测试")
    print("="*60)

    tests = [
        test_state_management,
        test_planner_node,
        test_tool_caller_node,
        test_critic_node,
        test_executor_node,
        test_workflow,
        test_convenience_function
    ]

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__doc__}: {e}")
        except Exception as e:
            print(f"  [ERROR] {test.__doc__}: {e}")

    print("\n" + "="*60)
    print("[OK] 所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
