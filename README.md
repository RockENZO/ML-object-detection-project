YOLOv8s four-class football detector (ball, goalkeeper, player, referee): ball-1280 fine-tuned for 25 more epochs at 1280 px with random blur, motion blur, noise and JPEG degradation on 40% of training images (run name ball-1280-robust, best epoch 25), optimizer state stripped.
Val mAP50 0.895, mAP50-95 0.658, ball mAP50 0.719, ball recall 0.605, ball precision 0.971 (validation split only; the frozen test split was not used). Inference size 1280.
Val mAP50 under perturbation (clean 0.895): blur 0.891, noise 0.890, JPEG q25 0.887, half resolution 0.894, dark 0.899, bright 0.886. The training degradations overlap these test perturbations, so this is an in-distribution stress test on one dataset, not evidence of generalisation to other leagues or cameras.
sha256(best.pt) = 5c5674f7775fd67b9ef6cd49a62ab8bc9aba5187bece9cc420897af5060ed3d3 (started from ball-1280, sha256 64556444238baca639ac1c435a63f5481b6d2cdda5215ee6b45c7814c2b8db21)
Licence: AGPL-3.0 (see the main branch LICENSE and NOTICE.md). Dataset: Roboflow football-players-detection, CC BY 4.0.
Use as INIT for fine-tuning; see docs/TRAINING.md on main. This branch only carries the weights file.
