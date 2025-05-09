import utils

from string import Template
from itertools import product


ARC_TEMPLATE = Template("$subj, who $vp1, $vp2.")
COORDINATION_TEMPLATE = Template("$subj $vp1 and $vp2.")

NO = ["No.", "That's not true.", "I doubt that.", "I don't think so."]
WAIT = ["Wait no.", "Hey, wait a minute.", "Hold on.", "Wait."]


def arc_template(subj, vp1, vp2, **kwargs):
    substituted = ARC_TEMPLATE.substitute(subj=subj, vp1=vp1, vp2=vp2)
    return substituted


def coordination(subj, vp1, vp2, **kwargs):
    substituted = COORDINATION_TEMPLATE.substitute(subj=subj, vp1=vp1, vp2=vp2)

    return substituted


def reject_sentences(verb1, verb2, prn, **kwargs):
    return f"{prn} {verb1} not.", f"{prn} {verb2} not."


def swap_item(item):
    return {
        "verb1": item["verb2"],
        "verb2": item["verb1"],
        "vp1": item["vp2"],
        "vp2": item["vp1"],
        "subj": item["subj"],
        "prn": item["prn"],
        "name1": item["name1"],
        "name2": item["name2"],
    }


items = utils.read_csv_dict("data/used_items.csv")

rejection_combos = list(product(NO, WAIT))

arcs = []
coords = []

idx = 1
for i, item in enumerate(items):
    swapped_item = swap_item(item)
    continuation1, continuation2 = reject_sentences(**item)
    continuation1, continuation2 = continuation1.capitalize(), continuation2.capitalize()
    arc = arc_template(**item)
    arc_swapped = arc_template(**swapped_item)
    coord = coordination(**item)
    coord_swapped = coordination(**swapped_item)

    for rc_id, rc in enumerate(rejection_combos):

        # generate one for normal
        # we want to return the sentence, the negation component, and the continuation
        arc_entry = {
            "idx": idx,
            "item": i + 1,
            "rejection_id": rc_id,
            "swapped": "False",
            "type": "arc",
            "name1": item["name1"],
            "name2": item["name2"],
            "preamble": arc,
            "no": rc[0],
            "wait": rc[1],
            "continuation1": continuation1,
            "continuation2": continuation2,
        }

        coord_entry = {
            "idx": idx,
            "item": i + 1,
            "rejection_id": rc_id,
            "swapped": "False",
            "type": "coord",
            "name1": item["name1"],
            "name2": item["name2"],
            "preamble": coord,
            "no": rc[0],
            "wait": rc[1],
            "continuation1": continuation1,
            "continuation2": continuation2,
        }

        # generate one for reversed

        arc_entry_swapped = {
            "idx": idx + 1,
            "item": i + 1,
            "rejection_id": rc_id,
            "swapped": "True",
            "type": "arc",
            "name1": swapped_item["name1"],
            "name2": swapped_item["name2"],
            "preamble": arc_swapped,
            "no": rc[0],
            "wait": rc[1],
            "continuation1": continuation2,
            "continuation2": continuation1,
        }

        coord_entry_swapped = {
            "idx": idx + 1,
            "item": i + 1,
            "rejection_id": rc_id,
            "swapped": "True",
            "type": "coord",
            "name1": swapped_item["name1"],
            "name2": swapped_item["name2"],
            "preamble": coord_swapped,
            "no": rc[0],
            "wait": rc[1],
            "continuation1": continuation2,
            "continuation2": continuation1,
        }

        idx += 2

        arcs.append(arc_entry)
        arcs.append(arc_entry_swapped)

        coords.append(coord_entry)
        coords.append(coord_entry_swapped)


utils.write_dict_list_to_csv(arcs, "data/stimuli/kim22-arc-dcpmi.csv")
utils.write_dict_list_to_csv(coords, "data/stimuli/kim22-coord-dcpmi.csv")
