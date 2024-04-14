import numpy as np

def computeFvalue(pv, rv, beta):
    temp = np.power(beta, 2)
    return (temp + 1) * pv * rv * 1.0 / (temp * pv + rv)


def Evaluation(p_c, groundEvent, s, numSnapshots, **kwargs):
    if len(p_c) == 0:
        print ( kwargs, "there is no point be detected...")
        return

    g_c = groundEvent

    inters = p_c.intersection(g_c)
    pg_n = len(inters)

    p_c = list(p_c)
    g_c = list(g_c)

    g_n = len(g_c)
    p_n = len(p_c)

    tempM = np.zeros((p_n, g_n))
    for i in range(p_n):
        for j in range(g_n):
            tempM[i, j] = np.abs((p_c[i] - g_c[j]))

    tempM = np.matrix(tempM)

    tempP = np.min(tempM, 1)
    tempP = tempP.reshape(1, tempP.shape[0]).tolist()[0]
    precision = 1.0 * len([item for item in tempP if item <= s]) / p_n

    tempR = np.min(tempM, 0)
    tempR = tempR.tolist()[0]
    recall = 1.0 * len([item for item in tempR if item <= s]) / g_n

    fpr = (p_n - pg_n) * 1.0 / (numSnapshots - g_n)
    fvalue = 0 if precision == 0 and recall == 0  else computeFvalue(precision,
                                                                     recall,
                                                                     1)
    return precision, recall, fvalue, fpr
