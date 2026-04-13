BASE_POINTS_PER_CELL = 1
CLEAR_POINTS_PER_CELL = 10


def calculate_placement_score(cells_placed: int) -> int:
    return cells_placed * BASE_POINTS_PER_CELL


def calculate_clear_score(lines_cleared: int, cells_cleared: int, combo: int) -> int:
    if cells_cleared == 0:
        return 0

    base_points = cells_cleared * CLEAR_POINTS_PER_CELL
    line_bonus_multiplier = 1 + (lines_cleared - 1) * 0.5 if lines_cleared >= 2 else 1.0
    combo_multiplier = 1 + combo

    return round(base_points * line_bonus_multiplier * combo_multiplier)
