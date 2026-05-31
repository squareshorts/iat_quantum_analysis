# Life-satisfaction IAT stimulus/theta audit

Key finding: the near-orthogonal theta in the life-satisfaction IAT is not explained by unequal item counts; each critical stimulus appears 180 times in the congruent phase and 180 times in the incongruent phase. It is strongly explained by the way item identity is used as the x-axis: base_trial 1-10 are logical true/false anchors and base_trial 11-20 are life-satisfaction statements, which are much slower. When display order is used instead of base_trial/item identity, theta moves from MAP 180.00 degrees to MAP 17.75 degrees.

Theta variants:
- collapsed_total_item_id_all: mean=177.48, MAP=180.00, 94% HDI=[172.75, 180.00], n=180
- collapsed_total_display_order_all: mean=28.49, MAP=17.75, 94% HDI=[8.50, 53.75], n=180
- first_attempt_item_id_all: mean=177.48, MAP=180.00, 94% HDI=[172.75, 180.00], n=180
- final_attempt_item_id_all: mean=177.48, MAP=180.00, 94% HDI=[172.75, 180.00], n=180
- collapsed_total_item_id_logical-anchor: mean=154.51, MAP=180.00, 94% HDI=[90.75, 180.00], n=180
- collapsed_total_display_order_logical-anchor: mean=25.34, MAP=16.25, 94% HDI=[8.00, 47.25], n=180
- first_attempt_item_id_logical-anchor: mean=154.51, MAP=180.00, 94% HDI=[90.75, 180.00], n=180
- collapsed_total_item_id_life-satisfaction statement: mean=163.05, MAP=180.00, 94% HDI=[128.50, 180.00], n=180
- collapsed_total_display_order_life-satisfaction statement: mean=35.44, MAP=17.50, 94% HDI=[7.50, 71.75], n=180
- first_attempt_item_id_life-satisfaction statement: mean=163.05, MAP=180.00, 94% HDI=[128.50, 180.00], n=180
- collapsed_total_item_id_error_free_items_only: mean=176.48, MAP=180.00, 94% HDI=[169.75, 180.00], n=180
- family_centered_within_participant_block_item_id_all: mean=42.02, MAP=18.50, 94% HDI=[7.25, 87.75], n=180

Stage by stimulus family:
- Congruent / life-satisfaction statement: n=1800, mean=3037.99 ms, median=2495.50 ms, IQR=1897.25 ms, errors=599/1800 (33.3%)
- Congruent / logical-anchor: n=1800, mean=1992.79 ms, median=1657.50 ms, IQR=1002.50 ms, errors=57/1800 (3.2%)
- Incongruent / life-satisfaction statement: n=1800, mean=3609.74 ms, median=2828.50 ms, IQR=2206.75 ms, errors=878/1800 (48.8%)
- Incongruent / logical-anchor: n=1800, mean=2546.22 ms, median=2088.00 ms, IQR=1379.75 ms, errors=1373/1800 (76.3%)

Item balance:
- Congruent item counts: min=180, max=180.
- Incongruent item counts: min=180, max=180.
- Mean item-level incongruent-minus-congruent RT difference=562.59 ms; range=236.37 to 1379.91 ms.
- Median item-level incongruent-minus-congruent RT difference=371.57 ms; range=-90.00 to 748.50 ms.

Leave-one-item-out:
- Dropping any one item leaves theta MAP at 180.00-180.00 degrees and theta mean at 175.93-177.64 degrees. This argues against a single problematic stimulus as the sole explanation.

Interpretation:
- There is no count imbalance by stimulus: the design is numerically balanced across phases.
- There is a strong stimulus-family/order confound: logical anchors occupy item IDs 1-10, life-satisfaction statements occupy item IDs 11-20, and life-satisfaction statements are about 1.0 s slower than logical anchors in both phases.
- The current theta pipeline treats base_trial/item ID as the within-block x coordinate. Therefore it captures item-family structure as if it were temporal structure.
- Using actual display order instead of item ID collapses the life-satisfaction theta into the same low-angle regime as the public IATs. This strongly supports the collaborator hypothesis in a refined form: the issue is not unequal counts, but stimulus-family/order imbalance in the x-axis used for profiling.