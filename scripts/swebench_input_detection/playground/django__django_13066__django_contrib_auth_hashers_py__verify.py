from django.contrib.auth.hashers import Argon2PasswordHasher


def test_input(pred_input):
	obj_ins = Argon2PasswordHasher()
	obj_ins_pred = Argon2PasswordHasher()
	assert obj_ins.verify(password = 'secret', encoded = 'argon2$argon2id$v=19$m=102400,t=2,p=8$Y041dExhNkljRUUy$TMa6A8fPJhCAUXRhJXCXdw')==obj_ins_pred.verify(password = pred_input['args']['password'], encoded = pred_input['args']['encoded']), 'Prediction failed!'
