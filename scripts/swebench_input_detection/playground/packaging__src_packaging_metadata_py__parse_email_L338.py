# Problem: packaging@@src_packaging_metadata.py@@parse_email_L338
# Module: packaging.metadata
# Function: parse_email
# Line: 338

from packaging.metadata import parse_email


def test_input(pred_input):
    assert parse_email(data = 'download-url: VaLuE')==parse_email(data = pred_input['args']['data']), 'Prediction failed!'
    
