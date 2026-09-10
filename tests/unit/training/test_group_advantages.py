from adaptive_math.training.reward_bridge import group_advantages


def test_group_advantages_are_normalized_and_constant_groups_are_ineffective() -> None:
    mixed = group_advantages([0.0, 1.0])
    constant = group_advantages([1.0, 1.0])

    assert mixed.effective
    assert mixed.advantages[0] < 0 < mixed.advantages[1]
    assert not constant.effective
    assert constant.advantages == (0.0, 0.0)
