import itertools
import json
import os
import re
import shutil
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from typing import List, Union
import numpy as np


MAPPING = {
    "id": "_id",
    "parsed_predict": "response"
}

def hook(dct):
    for k_old, k_new in MAPPING.items():
        if k_old in dct:
            dct[k_new] = dct[k_old]
            del dct[k_old]
    return dct


def load_json(path):
    return json.load(open(path, encoding='utf-8'))


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_jsonl(path, hook=None):
    res = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            res.append(json.loads(line, object_hook=hook))
    return res


def load_and_flatten_jsonl(path):
    res = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line, object_hook=hook)
            # flatten model response
            if "response" in sample:
                for model_response in sample["response"]:
                    tmp_sample = deepcopy(sample)
                    tmp_sample["response"] = model_response
                    res.append(tmp_sample)
            else:
                res.append(sample)
    return res


def save_jsonl(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')


def del_file(path):
    for elm in Path(path).glob('*'):
        elm.unlink() if elm.is_file() else shutil.rmtree(elm)
    if os.path.exists(path):
        os.rmdir(path)


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def change_match(file_p, language, natural_lang, ckpt_name):
    # change flags in field matched
    errors_p = f"data/temp/{language}_{natural_lang}_test/{ckpt_name}/error_problems.jsonl"
    response, errors  = load_jsonl(file_p), load_jsonl(errors_p)
    
    errors_dct = defaultdict(list)
    for row in errors:
        _, _id, response_num = row["problem"].split("_")
        errors_dct[int(_id)].append(int(response_num))
    
    for row in response:
        row["matched"] = [True] * len(row["matched"])
        for idx, _ in enumerate(row["matched"]):
            if idx in errors_dct[row["id"]]:
                row["matched"][idx] = False
    
    save_jsonl(response, file_p)
    
    # change accuracy in all_metrics.json
    result = load_json(f"results/{ckpt_name}/result.json")
    all_metrics = load_jsonl(f"result/{ckpt_name}/all_metrics.jsonl")
    
    updated_accuracy = result["result"]["pass@k"]["pass@1"]
    for metric in all_metrics:
        if language == "java" and re.search("java", metric["dataset"], flags=re.I) is not None:
            metric["metric"]["accuracy@1"] = updated_accuracy
    
        if language == "python" and re.search("py", metric["dataset"], flags=re.I) is not None:
            metric["metric"]["accuracy@1"] = updated_accuracy
            
    save_jsonl(all_metrics, f"result/{ckpt_name}/all_metrics.jsonl")
