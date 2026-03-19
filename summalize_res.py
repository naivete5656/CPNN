import pandas as pd
import numpy as np

from main import get_parse
from utils import naming_function


if __name__ == "__main__":
    args = get_parse()

    pcc_list, scc_list = [], []
    for fold in range(4):
        args.fold = fold
        run_name = naming_function(args)

        resutl_path = f"outputs/{fold}/{run_name}.txt"
        if "ST" in args.trainer:
            pcc, scc = pd.read_csv(resutl_path).values[0][1:3]
        else:
            pcc, scc = pd.read_csv(resutl_path).values[0][2:4]
        pcc_list.append(pcc)
        scc_list.append(scc)
    pcc = np.array(pcc_list).mean()
    scc = np.array(scc_list).mean()

    # save results
    df = pd.DataFrame({"fold": list(range(4)), "PCC": pcc_list, "SCC": scc_list})
    df.loc["mean"] = ["mean", pcc, scc]
    df.to_csv(f"outputs/{run_name}_summary.csv", index=False)
