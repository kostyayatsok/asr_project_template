import editdistance

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    n = len(target_text)
    if n == 0: return 0.
    return editdistance.eval(predicted_text.lower(), target_text.lower()) / n


def calc_wer(target_text, predicted_text) -> float:
    n = len(target_text.split())
    if n == 0: return 0.
    return editdistance.eval(predicted_text.lower().split(), target_text.lower().split()) / n

# import numpy as np

# for target, pred, expected_wer, expected_cer in [
#     ("if you can not measure it you can not improve it", 
#      "if you can nt measure t yo can not i", 
#      0.454, 0.25),
#     ("if you cant describe what you are doing as a process you dont know what youre doing", 
#      "if you cant describe what you are doing as a process you dont know what youre doing", 
#      0.0, 0.0),
#     ("one measurement is worth a thousand expert opinions", 
#      "one  is worth thousand opinions", 
#      0.375, 0.392)
# ]:
#     wer = calc_wer(target, pred)
#     cer = calc_cer(target, pred)
#     assert np.isclose(wer, expected_wer, atol=1e-3), f"true: {target}, pred: {pred}, expected wer {expected_wer} != your wer {wer}"
#     assert np.isclose(cer, expected_cer, atol=1e-3), f"true: {target}, pred: {pred}, expected cer {expected_cer} != your cer {cer}"

#     print('OK')