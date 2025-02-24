import pandas as pd

def remove_empty_rows(frame: pd.DataFrame, label) -> pd.DataFrame:
    match label:
        case 'lab':
            return frame[frame.lab != '']
        case 'lcn':
            return frame[frame.lcn != '']
        case 'lpn':
            return frame[frame.lnp != '']
        case 'sbj':
            return frame[frame.sbj != '']
        case 'dsc':
            return frame[frame.dsc != '']
        case 'curi':
            return frame[frame.curi != '']
        case 'puri':
            return frame[frame.puri != '']
        case 'voc':
            return frame[frame.voc != '']
        case 'tdl':
            return frame[frame.tdl != '']
        case _:
            return frame



