import os
import utils
import pathlib

import numpy as np

from collections import defaultdict
from copy import deepcopy

kim_arc = utils.read_csv_dict("data/stimuli/kim22-arc-unique.csv")
kim_coord = utils.read_csv_dict("data/stimuli/kim22-coord-unique.csv")

kim_arc_id_wise = defaultdict(list)
for entry in kim_arc:
    kim_arc_id_wise[int(entry["item"])].append(entry)

kim_arc_id_wise = dict(kim_arc_id_wise)

kim_coord_id_wise = defaultdict(list)
for entry in kim_coord:
    kim_coord_id_wise[int(entry["item"])].append(entry)

kim_coord_id_wise = dict(kim_coord_id_wise)


def read_preprocess(path):
    generations = utils.read_json(path)
    generations_1 = defaultdict(list)
    generations_2 = defaultdict(list)

    for item in generations["generation_vp1"]:
        generations_1[item["idx"]].extend(item["sentences"])
    for item in generations["generation_vp2"]:
        generations_2[item["idx"]].extend(item["sentences"])

    # return generations_1, generations_2
    return {"vp1": generations_1, "vp2": generations_2}


def process_gens(collection, path="", sample=10):
    """
    item, continuation_type (vp1/vp2), continuation_id, continuation
    """
    content = []
    for vp_type, gens in collection.items():
        for idx, continuations in gens.items():
            sampled = continuations[:sample]
            for i, (sentence, logprob) in enumerate(sampled):
                content.append((idx, vp_type, i + 1, sentence.strip()))
    
    return content
    # utils.write_csv(
    #     content,
    #     path,
    #     header=["item", "continuation_type", "continuation_id", "continuation"],
    # )


PATH = "data/results/generations"

# model_results = defaultdict(dict)

for dir in os.listdir(PATH):
    # freeform = defaultdict(list)
    # rejection = defaultdict(list)
    freeform = defaultdict(lambda: defaultdict(set))
    rejection = defaultdict(lambda: defaultdict(set))
    generation_dir = f"{PATH}/{dir}/"
    for file in os.listdir(generation_dir):
        if "json" in file:
            full_path = f"{generation_dir}/{file}"
            preprocessed = read_preprocess(full_path)
            if "freeform" in file:
                for k, v in preprocessed.items():
                    for idx, gens in v.items():
                        for g in gens:
                            freeform[k][idx].add(tuple(g))
            elif "rejection" in file:
                for k, v in preprocessed.items():
                    for idx, gens in v.items():
                        for g in gens:
                            rejection[k][idx].add(tuple(g))

    # freeform = {k: {kk: sorted(vv, key=lambda x: -x[-1])} for kk, vv in freeform.items()}
    freeform = {
        k: {kk: sorted(vv, key=lambda x: -x[-1]) for kk, vv in v.items()}
        for k, v in freeform.items()
    }

    rejection = {
        k: {kk: sorted(vv, key=lambda x: -x[-1]) for kk, vv in v.items()}
        for k, v in rejection.items()
    }

    model_results = {"freeform": freeform, "rejection": rejection}

    for k, v in model_results.items():
        save_path = f"data/results/sorted-generations/{k}"
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

        # write_gens(v, f"{save_path}/{dir}.csv")
        processed = process_gens(v)
        arc_stimuli = []

        for entry in processed:
            arc_entries = kim_arc_id_wise[entry[0]]
            for ae in arc_entries:
                item = deepcopy(ae)
                item.update({
                    'continuation_type': entry[1],
                    'continuation_id': entry[2],
                    'continuation': entry[3]
                })
                arc_stimuli.append(item)

        utils.write_dict_list_to_csv(arc_stimuli, f"{save_path}/{dir}-arc.csv")

        coord_stimuli = []

        for entry in processed:
            coord_entries = kim_coord_id_wise[entry[0]]
            for ce in coord_entries:
                item = deepcopy(ce)
                item.update({
                    'continuation_type': entry[1],
                    'continuation_id': entry[2],
                    'continuation': entry[3]
                })
                coord_stimuli.append(item)

        print(f"Model: {dir}. Len: {len(arc_stimuli)}")

        utils.write_dict_list_to_csv(coord_stimuli, f"{save_path}/{dir}-coord.csv")
