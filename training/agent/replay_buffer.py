import numpy as np


class SumTree:
    """Binary tree where each parent is the sum of its children.
    Leaf nodes store transition priorities. Enables O(log n) proportional sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        idx = self.data_pointer
        tree_idx = self.data_pointer + self.capacity - 1
        self._update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

    def _update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx

    def priority(self, data_idx: int) -> float:
        return float(self.tree[data_idx + self.capacity - 1])


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 500_000, alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.states = np.zeros((capacity, 7, 8, 8), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 7, 8, 8), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, 192), dtype=bool)

        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ):
        idx = self.tree.add(self._max_priority ** self.alpha)
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_masks[idx] = next_action_mask

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        indices: list[int] = []
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.get(value)
            idx = max(0, min(idx, len(self) - 1))
            indices.append(idx)
            priorities[i] = self.tree.priority(idx)

        priorities = np.clip(priorities, self.epsilon, None)
        probs = priorities / self.tree.total()
        weights = (len(self) * probs) ** (-beta)
        weights = weights / weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_masks[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, abs(float(td_error)) + self.epsilon)
