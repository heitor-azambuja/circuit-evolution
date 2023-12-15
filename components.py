import logging
logger = logging.getLogger()


class DigitalPot:
	def __init__(self, max_value, tap_points) -> None:
		self.max_value = max_value
		self.tap_points = tap_points
		self.min_value = max_value / (tap_points - 1)


	def get_resistance(self, position) -> float:
		if position < 1:
			logger.warning(f'Position must be between 1 and {self.tap_points}. setting position equal to 1')
			position = 1

		if position > self.tap_points:
			logger.warning(f'Position must be between 1 and {self.tap_points}. setting position equal to {self.tap_points}')
			position = self.tap_points

		return self.min_value * (position - 1)