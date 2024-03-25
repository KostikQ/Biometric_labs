import numpy as np
import cv2
from finger_print import FingerPrint
from tqdm import tqdm

def compareFingerPrints(example_path: str, fingerprint_path: str) -> None:

    e_image = cv2.imread(example_path)
    c_image = cv2.imread(fingerprint_path)
    cv2.imshow("example", e_image)
    cv2.imshow("compared", c_image)

    example = FingerPrint(e_image)
    example()
    example.draw_detection()    
    
    compared = FingerPrint(c_image)
    compared()
    compared.draw_detection()
    
    n_prob = 0
    t_prob = 0
    count = 0
    t_sum_prob = min(np.sum(example.t_count), np.sum(compared.t_count))/max(np.sum(example.t_count), np.sum(compared.t_count))
    n_sum_prob = min(np.sum(example.n_count), np.sum(compared.n_count))/max(np.sum(example.n_count), np.sum(compared.n_count))

    for row_ind in tqdm(range(len(compared.n_count))):
        for col_ind in range(len(compared.n_count[row_ind])):
            count += 1
            if compared.t_count[row_ind, col_ind] == example.t_count[row_ind, col_ind]:
                t_prob +=1
            else:
                t_prob += min(compared.t_count[row_ind, col_ind], example.t_count[row_ind, col_ind]) / max(compared.t_count[row_ind, col_ind], example.t_count[row_ind, col_ind]) 
            if compared.n_count[row_ind, col_ind] == example.n_count[row_ind, col_ind]:
                n_prob +=1
            else:
                n_prob += min(compared.n_count[row_ind, col_ind], example.n_count[row_ind, col_ind]) / max(compared.n_count[row_ind, col_ind], example.n_count[row_ind, col_ind])
    n_prob /= count
    t_prob /= count
    
    return t_sum_prob, n_sum_prob, t_prob, n_prob