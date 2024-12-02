"""
Script to generate and save stimuli. Multiple modes:

1. kim22
    - The original stimuli from Kim et al. 2022.
    - right now, allows computation of full stimulus log probabilities, as well as:
        p("verb not." | "prefix + No/Wait no")
2. manual (under construction)
    - manually created stimuli by Sanghee
"""

import argparse
import pathlib
import utils

from string import Template

TEMPLATE = Template(
    '$name1 said, "$subj, who $arc, $mc." $name2 replied, "$negation,[SPLIT]$prn $mainverb not."'
)


def generate_kim22_stimuli(name1, subj, vps, name2, prn, verbs):
    """vps formatted as vp1, vp2"""
    neg_main_good = TEMPLATE.substitute(
        name1=name1,
        subj=subj,
        arc=vps[0],
        mc=vps[1],
        name2=name2,
        negation="No",
        prn=prn,
        mainverb=verbs[1],
    )
    neg_main_bad = TEMPLATE.substitute(
        name1=name1,
        subj=subj,
        arc=vps[0],
        mc=vps[1],
        name2=name2,
        negation="Wait no",
        prn=prn,
        mainverb=verbs[1],
    )
    neg_arc_good = TEMPLATE.substitute(
        name1=name1,
        subj=subj,
        arc=vps[0],
        mc=vps[1],
        name2=name2,
        negation="Wait no",
        prn=prn,
        mainverb=verbs[0],
    )
    neg_arc_bad = TEMPLATE.substitute(
        name1=name1,
        subj=subj,
        arc=vps[0],
        mc=vps[1],
        name2=name2,
        negation="No",
        prn=prn,
        mainverb=verbs[0],
    )

    return neg_main_good, neg_main_bad, neg_arc_good, neg_arc_bad


def main(args):

    mode = args.mode
    item_path = args.item_path
    save_dir = args.save_dir

    eval_data = []
    if mode == "kim22":
        # kim, lang, ettinger (2022)

        eval_items = utils.read_csv_dict(item_path)
        idx = 0
        for i, item in enumerate(eval_items):
            # verb2 and vp2 are main
            neg_main_good, neg_main_bad, neg_arc_good, neg_arc_bad = (
                generate_kim22_stimuli(
                    item["name1"],
                    item["subj"],
                    [item["vp1"], item["vp2"]],
                    item["name2"],
                    item["prn"],
                    [item["verb1"], item["verb2"]],
                )
            )
            # following tidy principles - each entry is one item!
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_main_good.replace("[SPLIT]", " "),
                    "prefix": neg_main_good.split("[SPLIT]")[0],
                    "stimulus": neg_main_good.split("[SPLIT]")[1],
                    "type": "neg_main_good",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_main_bad.replace("[SPLIT]", " "),
                    "prefix": neg_main_bad.split("[SPLIT]")[0],
                    "stimulus": neg_main_bad.split("[SPLIT]")[1],
                    "type": "neg_main_bad",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_arc_good.replace("[SPLIT]", " "),
                    "prefix": neg_arc_good.split("[SPLIT]")[0],
                    "stimulus": neg_arc_good.split("[SPLIT]")[1],
                    "type": "neg_arc_good",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_arc_bad.replace("[SPLIT]", " "),
                    "prefix": neg_arc_bad.split("[SPLIT]")[0],
                    "stimulus": neg_arc_bad.split("[SPLIT]")[1],
                    "type": "neg_arc_bad",
                }
            )

            # reverse -- verb1 and vp1 are main
            idx += 1
            neg_main_good, neg_main_bad, neg_arc_good, neg_arc_bad = (
                generate_kim22_stimuli(
                    item["name1"],
                    item["subj"],
                    [item["vp2"], item["vp1"]],
                    item["name2"],
                    item["prn"],
                    [item["verb2"], item["verb1"]],
                )
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_main_good.replace("[SPLIT]", " "),
                    "prefix": neg_main_good.split("[SPLIT]")[0],
                    "stimulus": neg_main_good.split("[SPLIT]")[1],
                    "type": "neg_main_good",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_main_bad.replace("[SPLIT]", " "),
                    "prefix": neg_main_bad.split("[SPLIT]")[0],
                    "stimulus": neg_main_bad.split("[SPLIT]")[1],
                    "type": "neg_main_bad",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_arc_good.replace("[SPLIT]", " "),
                    "prefix": neg_arc_good.split("[SPLIT]")[0],
                    "stimulus": neg_arc_good.split("[SPLIT]")[1],
                    "type": "neg_arc_good",
                }
            )
            eval_data.append(
                {
                    "idx": idx,
                    "item": i,
                    "sentence": neg_arc_bad.replace("[SPLIT]", " "),
                    "prefix": neg_arc_bad.split("[SPLIT]")[0],
                    "stimulus": neg_arc_bad.split("[SPLIT]")[1],
                    "type": "neg_arc_bad",
                }
            )
            idx += 1

        print(neg_arc_good)

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        utils.write_dict_list_to_csv(eval_data, f"{save_dir}/kim22-full.csv")
        

    elif mode == "manual":
        """For the stimuli that Sanghee creates manually!"""
        pass
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="kim22")
    parser.add_argument("--item-path", type=str, default="data/used_items.csv")
    parser.add_argument("--save-dir", type=str, default="data/stimuli")

    args = parser.parse_args()
    main(args)
