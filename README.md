YOLOv8s four-class football detector (ball, goalkeeper, player, referee): the 72-epoch detector fine-tuned for 40 epochs at 960 px on the grouped training split (run name ball-960), optimizer state stripped.
Val mAP50 0.864, mAP50-95 0.641, ball mAP50 0.586, ball recall 0.432 (validation split only; the frozen test split was not used).
sha256(best.pt) = b5676be890faf7b864a6be31bce56f9d644ba69fedc567f27c4d6b86b6e26acf (started from sha256 29e7af46ed5f5928d9a17ae20e287fee85ddb26df34e201013f0c93cd4ecd1f5)
Licence: AGPL-3.0 (see the main branch LICENSE and NOTICE.md). Dataset: Roboflow football-players-detection, CC BY 4.0.
Use as INIT for fine-tuning; see docs/TRAINING.md on main. This branch only carries the weights file.
