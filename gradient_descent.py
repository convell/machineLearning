import numpy as  np

def gradient_descent(grad_f,x_init,learning_rate):
	grad_is_zero_flag = 1
	counter = 0

	while grad_is_zero_flag >= 1:
		grad_is_zero_flag = 0

		grad_value = grad_f(x_init)

		for sub_value in grad_value:
			if sub_value >= 0.0001:
				grad_is_zero_flag += 1

		if grad_is_zero_flag >= 1:
			x_init = x_init - learning_rate*(grad_value)
			counter += 1

	return x_init