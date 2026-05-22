import numpy as np


class State:
    """
    Состояние позиции для contextual bandit.
    """

    def __init__(self):
        self.in_position = 0
        self.entry_price = None
        self.max_price = None

        self.bars_in_position = 0
        self.time_since_last_trade = 10_000
        self.last_action = 0

    def on_bar(self, current_price):
        self.time_since_last_trade += 1

        if self.in_position == 1:
            self.bars_in_position += 1
            self.max_price = max(self.max_price, current_price)

    def enter(self, current_price):
        self.in_position = 1
        self.entry_price = current_price
        self.max_price = current_price

        self.bars_in_position = 0
        self.time_since_last_trade = 0
        self.last_action = 1

    def exit(self):
        self.in_position = 0
        self.entry_price = None
        self.max_price = None

        self.bars_in_position = 0
        self.time_since_last_trade = 0
        self.last_action = 0

    def context(self, current_price):
        if self.in_position == 0:
            unrealized_pnl = 0.0
            drawdown = 0.0
        else:
            unrealized_pnl = np.log(current_price / self.entry_price)
            drawdown = (self.max_price - current_price) / (self.max_price + 1e-9)

        return np.array([
            self.in_position,
            np.log1p(self.bars_in_position),
            unrealized_pnl,
            drawdown,
            np.log1p(self.time_since_last_trade),
            self.last_action,
        ], dtype=np.float64)