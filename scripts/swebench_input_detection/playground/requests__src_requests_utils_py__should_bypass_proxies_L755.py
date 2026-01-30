# Problem: requests@@src_requests_utils.py@@should_bypass_proxies_L755
# Module: requests.utils
# Function: should_bypass_proxies
# Line: 755

from requests.utils import should_bypass_proxies


def test_input(pred_input):
    assert should_bypass_proxies(url = 'http://localhost:48329/', no_proxy = None)==should_bypass_proxies(url = pred_input['args']['url'], no_proxy = pred_input['args']['no_proxy']), 'Prediction failed!'
    
    
