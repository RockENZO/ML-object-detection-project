YOLOv8s four-class football detector (ball, goalkeeper, player, referee): the ball-960 detector fine-tuned for 30 epochs at 1280 px (batch 4, cosine learning rate, stronger augmentation; run name ball-1280, best epoch 21), optimizer state stripped.
Val mAP50 0.886, mAP50-95 0.657, ball mAP50 0.676, ball recall 0.547, ball precision 0.984 (validation split only; the frozen test split was not used). Inference size 1280.
sha256(best.pt) = 64556444238baca639ac1c435a63f5481b6d2cdda5215ee6b45c7814c2b8db21 (started from ball-960, sha256 b5676be890faf7b864a6be31bce56f9d644ba69fedc567f27c4d6b86b6e26acf, itself fine-tuned from 29e7af46ed5f5928d9a17ae20e287fee85ddb26df34e201013f0c93cd4ecd1f5)
Licence: AGPL-3.0 (see the main branch LICENSE and NOTICE.md). Dataset: Roboflow football-players-detection, CC BY 4.0.
Use as INIT for fine-tuning; see docs/TRAINING.md on main. This branch only carries the weights file.
