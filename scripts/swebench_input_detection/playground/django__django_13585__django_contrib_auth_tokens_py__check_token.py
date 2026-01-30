import django
from django.conf import settings

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
    SECRET_KEY='dummy'
)

from django.contrib.auth.tokens import PasswordResetTokenGenerator

def test_input(pred_input):
	obj_ins = PasswordResetTokenGenerator()
	obj_ins_pred = PasswordResetTokenGenerator()
	assert obj_ins.check_token(user = {'_state': {'db': 'default', 'adding': False}, 'id': 1, 'password': 'md5$YbSnGA0Tetrj$704b247ccdb4be0f0f5df96a453b648c', 'last_login': None, 'is_superuser': False, 'username': 'tokentestuser', 'first_name': '', 'last_name': '', 'email': 'test2@example.com', 'is_staff': False, 'is_active': True, 'date_joined': '2025-04-22 16:19:07.249701'}, token = 'coo5bv-92cebe54064db62bd060fb49891f8efe')==obj_ins_pred.check_token(user = pred_input['args']['user'], token = pred_input['args']['token']), 'Prediction failed!'