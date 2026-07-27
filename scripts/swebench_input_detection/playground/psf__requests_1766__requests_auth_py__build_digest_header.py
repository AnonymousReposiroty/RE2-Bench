# Problem: psf__requests-1766@@requests.auth.py@@build_digest_header
# Benchmark: Swebench
# Module: requests.auth
# Function: build_digest_header

from requests.auth import HTTPDigestAuth


def test_input(pred_input):
    obj_ins = HTTPDigestAuth(username = 'user', password = 'pass')
    obj_ins.last_nonce = ''
    obj_ins.nonce_count = 0
    obj_ins.chal = {'realm': 'me@kennethreitz.com', 'nonce': '2cb6ca3bd92f46300c21bc6652450899', 'qop': 'auth', 'opaque': '3b88561378508443a8b14a91ec359904', 'algorithm': 'MD5', 'stale': 'FALSE'}
    obj_ins.pos = None
    obj_ins.num_401_calls = 2
    obj_ins_pred = HTTPDigestAuth(username = pred_input['self']['username'], password = pred_input['self']['password'])
    obj_ins_pred.last_nonce = pred_input['self']['last_nonce']
    obj_ins_pred.nonce_count = pred_input['self']['nonce_count']
    obj_ins_pred.chal = pred_input['self']['chal']
    obj_ins_pred.pos = pred_input['self']['pos']
    obj_ins_pred.num_401_calls = pred_input['self']['num_401_calls']
    assert obj_ins.build_digest_header(method = 'GET', url = 'http://httpbin.org/digest-auth/auth/user/pass')==obj_ins_pred.build_digest_header(method = pred_input['args']['method'], url = pred_input['args']['url']), 'Prediction failed!'
    