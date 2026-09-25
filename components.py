import logging
logger = logging.getLogger()


class DigitalPot:
	def __init__(self, max_value, tap_points, wiper_resistance=0.0) -> None:
		'''
			wiper_resistance is in series with the tapped portion when the part is
			wired as a rheostat, which is how these circuits use it. The X9C
			datasheet gives 40 ohm typical, 100 ohm max — on a 1k part that is
			four tap steps, so it is not a rounding detail. It defaults to zero so
			existing results stay comparable; pass it explicitly to model the part
			faithfully.
		'''
		self.max_value = max_value
		self.tap_points = tap_points
		self.wiper_resistance = wiper_resistance
		# The array is tap_points-1 resistive elements (99 for the X9C).
		self.min_value = max_value / (tap_points - 1)


	def get_resistance(self, position) -> float:
		if position < 1:
			logger.warning(f'Position must be between 1 and {self.tap_points}. setting position equal to 1')
			position = 1

		if position > self.tap_points:
			logger.warning(f'Position must be between 1 and {self.tap_points}. setting position equal to {self.tap_points}')
			position = self.tap_points

		return self.min_value * (position - 1) + self.wiper_resistance