def accuracy(y_true, y_pred):
	"""Calcula accuracy simple."""
	return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
