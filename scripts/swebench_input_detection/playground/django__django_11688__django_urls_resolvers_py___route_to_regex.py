from django.urls.resolvers import _route_to_regex

def test_input(pred_input):
    a = _route_to_regex(route = '"<lang>"/"<path:url>"/', is_endpoint = True)
    b = _route_to_regex(route = pred_input['args']['route'], is_endpoint = pred_input['args']['is_endpoint'])
    assert _route_to_regex(route = '"<lang>"/"<path:url>"/', is_endpoint = True)==_route_to_regex(route = pred_input['args']['route'], is_endpoint = pred_input['args']['is_endpoint']), 'Prediction failed!'