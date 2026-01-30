from seaborn._stats.regression import PolyFit
from seaborn._core.groupby import GroupBy
import pandas as pd
def test_input(pred_input):
    obj_ins = PolyFit(gridsize=100)
    df = pd.DataFrame({"x":[], "y":[]})
    obj_ins_pred = PolyFit(gridsize = pred_input['self']['gridsize'])
    assert obj_ins._call(data=df, groupby=GroupBy, orient="x", scales={})==obj_ins_pred._call(data = df, groupby=GroupBy, orient=pred_input['args']['orient'], scales=pred_input['args']['scales']), 'Prediction failed!'