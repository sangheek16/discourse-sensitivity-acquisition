## Brainstorm

RQs:

1. Does the model have a recency bias?
2. Does the model have an at-issue-only bias?
2-precise: Does the model display sensitivity to at-issueness?

Subj VP1 generations [g1_1, g1_2, ...]
Subj VP2 generations [g2_1, g2_2, ...]

sentence: ARC or RRC
Sent_arc: Subj, VP1, VP2.
<!--Sent_rrc: Subj VP1 VP2.-->
Sent_coord: Subj VP1 and VP2.

Eventually we want to address RQ2

Addressing 1:

When s = Sent_arc:
genscore_vp1 = log P(g1 | s)
genscore_vp2 = log P(g2 | s)

When s = Sent_coord:
genscore_vp1 = log P(g1 | s)
genscore_vp2 = log P(g2 | s)

if genscore_vp1 - genscore_vp2 < 0  for s = Sent_arc:
- either the model has a recency bias, or
- the model has an at-issue bias

to tease this apart, we consider s = Sent_coord. Here:
if genscore_vp1 - genscore_vp2 < 0, or if this difference is comparable to the difference in s = Sent_arc, then this is evidence in favor of recency bias
    
if genscore_vp1 - genscore_vp2 ~ 0, then this weakens the evidence for recency bias

For both cases (Sent_arc and Sent_coord), if we instead see genscore_vp1 - genscore_vp2 > 0, then this would suggest the models have neither recency nor at-issue only bias

Be upfront about interpretation problems if we observe vp1 bias for Sent_coord


---


Subj VP1 rejection generations [rg1_1, rg1_2, ...]
Subj VP2 rejection generations [rg2_1, rg2_2, ...]

Sent_arc: Subj, VP1, VP2.

Sent_arc. {No} rg1
Sent_arc. {No} rg2
Sent_arc. {Hey wait a min} rg1
Sent_arc. {Hey wait a min} rg2

1_2: 51 - 49 > 0 (requires DC-PMI) 
2_2: (49 - 48) > (51 - 50)


