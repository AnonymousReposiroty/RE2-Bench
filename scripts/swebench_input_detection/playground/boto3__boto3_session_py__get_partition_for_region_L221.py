# Problem: boto3@@boto3_session.py@@get_partition_for_region_L221
# Module: boto3.session
# Function: get_partition_for_region
# Line: 221

from boto3.session import Session


def test_input(pred_input):
    obj_ins = Session()
    obj_ins._session = "Session(region_name=<Mock name='mock.get_config_variable()' id='127141585996944'>)"
    obj_ins.resource_factory = ''
    obj_ins._loader = ''
    obj_ins_pred = Session()
    obj_ins_pred._session = pred_input['self']['_session']
    obj_ins_pred.resource_factory = pred_input['self']['resource_factory']
    obj_ins_pred._loader = pred_input['self']['_loader']
    assert obj_ins.get_partition_for_region(region_name = 'foo-bar-1')==obj_ins_pred.get_partition_for_region(region_name = pred_input['args']['region_name']), 'Prediction failed!'
    
