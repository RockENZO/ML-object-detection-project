YOLOv8s four-class football detector (ball, goalkeeper, player, referee), stopped early at epoch 72 (best epoch 52), optimizer state stripped.
Val mAP50 0.751, mAP50-95 0.508, ball mAP50 0.33, ball recall 0.16.
sha256(best.pt) = 29e7af46ed5f5928d9a17ae20e287fee85ddb26df34e201013f0c93cd4ecd1f5
Licence: AGPL-3.0 (see the main branch LICENSE and NOTICE.md). Dataset: Roboflow football-players-detection, CC BY 4.0.
Use as INIT for fine-tuning; see docs/TRAINING.md on main. This branch only carries the weights file.
