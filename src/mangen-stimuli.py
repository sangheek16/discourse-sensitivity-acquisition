import utils

from collections import defaultdict
from string import Template

ARC_TEMPLATE = Template("$subj, who $vp1, $vp2.")
COORDINATION_TEMPLATE = Template("$subj $vp1 and $vp2.")


def arc_template(subj, vp1, vp2, **kwargs):
    substituted = ARC_TEMPLATE.substitute(subj=subj, vp1=vp1, vp2=vp2)
    return substituted


def coordination(subj, vp1, vp2, **kwargs):
    substituted = COORDINATION_TEMPLATE.substitute(subj=subj, vp1=vp1, vp2=vp2)
    return substituted


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


mangen = utils.read_csv_dict("data/manual-items.csv")

unique_arcs = []
unique_coords = []

idx = 1
for i, item in enumerate(mangen):
    swapped_item = swap_item(item)
    arc = arc_template(**item)
    arc_swapped = arc_template(**swapped_item)
    coord = coordination(**item)
    coord_swapped = coordination(**swapped_item)

    # unique stimuli

    arc_entry = {
        "item": item["item"],
        "swapped": "False",
        "type": "arc",
        "name1": item["name1"],
        "name2": item["name2"],
        "preamble": arc,
        "continuation_id": item["continuation_id"],
        "continuation_class": item["continuation_type"],
        "continuation_type": f"vp{item['vp']}",
        "continuation": item["continuation"],
    }

    coord_entry = {
        "item": item["item"],
        "swapped": "False",
        "type": "coord",
        "name1": item["name1"],
        "name2": item["name2"],
        "preamble": coord,
        "continuation_id": item["continuation_id"],
        "continuation_class": item["continuation_type"],
        "continuation_type": f"vp{item['vp']}",
        "continuation": item["continuation"],
    }

    # generate one for reversed

    arc_entry_swapped = {
        "item": item["item"],
        "swapped": "True",
        "type": "arc",
        "name1": swapped_item["name1"],
        "name2": swapped_item["name2"],
        "preamble": arc_swapped,
        "continuation_id": item["continuation_id"],
        "continuation_class": item["continuation_type"],
        "continuation_type": f"vp{item['vp']}",
        "continuation": item["continuation"],
    }

    coord_entry_swapped = {
        "item": item["item"],
        "swapped": "True",
        "type": "coord",
        "name1": swapped_item["name1"],
        "name2": swapped_item["name2"],
        "preamble": coord_swapped,
        "continuation_id": item["continuation_id"],
        "continuation_class": item["continuation_type"],
        "continuation_type": f"vp{item['vp']}",
        "continuation": item["continuation"],
    }

    unique_arcs.append(arc_entry)
    unique_arcs.append(arc_entry_swapped)
    unique_coords.append(coord_entry)
    unique_coords.append(coord_entry_swapped)


utils.write_dict_list_to_csv(unique_arcs, "data/stimuli/mangen-arc.csv")
utils.write_dict_list_to_csv(unique_coords, "data/stimuli/mangen-coord.csv")