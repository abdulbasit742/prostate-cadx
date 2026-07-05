import json
import os

skills = [
    # GROUP A
    {"id": 1, "name": "detect_network", "group": "A", "status": "pending", "deps": []},
    {"id": 2, "name": "setup_venv", "group": "A", "status": "pending", "deps": ["detect_network"]},
    {"id": 3, "name": "install_torch_cuda", "group": "A", "status": "pending", "deps": ["setup_venv"]},
    {"id": 4, "name": "verify_gpu", "group": "A", "status": "pending", "deps": ["install_torch_cuda"]},
    {"id": 5, "name": "install_deps", "group": "A", "status": "pending", "deps": ["setup_venv"]},
    {"id": 6, "name": "init_repo", "group": "A", "status": "pending", "deps": []},
    {"id": 7, "name": "config_loader", "group": "A", "status": "pending", "deps": []},
    {"id": 8, "name": "logging_setup", "group": "A", "status": "pending", "deps": []},
    {"id": 9, "name": "ollama_client", "group": "A", "status": "pending", "deps": ["detect_network"]},
    {"id": 10, "name": "smoke_harness", "group": "A", "status": "pending", "deps": ["config_loader", "logging_setup"]},

    # GROUP B
    {"id": 11, "name": "kaggle_auth", "group": "B", "status": "pending", "deps": ["config_loader"]},
    {"id": 12, "name": "download_panda", "group": "B", "status": "pending", "deps": ["kaggle_auth", "detect_network"]},
    {"id": 13, "name": "download_gleason2019", "group": "B", "status": "pending", "deps": ["detect_network"]},
    {"id": 14, "name": "download_tcga_prad", "group": "B", "status": "pending", "deps": ["detect_network"]},
    {"id": 15, "name": "synthetic_dataset", "group": "B", "status": "pending", "deps": ["config_loader"]},
    {"id": 16, "name": "dataset_manifest", "group": "B", "status": "pending", "deps": ["synthetic_dataset"]},
    {"id": 17, "name": "checksum_verify", "group": "B", "status": "pending", "deps": ["dataset_manifest"]},
    {"id": 18, "name": "dataset_stats", "group": "B", "status": "pending", "deps": ["dataset_manifest"]},
    {"id": 19, "name": "train_val_split", "group": "B", "status": "pending", "deps": ["dataset_manifest"]},
    {"id": 20, "name": "data_registry", "group": "B", "status": "pending", "deps": ["dataset_manifest"]},

    # GROUP C
    {"id": 21, "name": "wsi_reader", "group": "C", "status": "pending", "deps": ["install_deps"]},
    {"id": 22, "name": "tissue_mask", "group": "C", "status": "pending", "deps": ["wsi_reader"]},
    {"id": 23, "name": "tile_extractor", "group": "C", "status": "pending", "deps": ["tissue_mask"]},
    {"id": 24, "name": "tile_filter", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 25, "name": "tile_label_map", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 26, "name": "stain_normalize_macenko", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 27, "name": "stain_normalize_reinhard", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 28, "name": "tile_cache", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 29, "name": "tile_dataset_class", "group": "C", "status": "pending", "deps": ["tile_extractor"]},
    {"id": 30, "name": "augment_pipeline", "group": "C", "status": "pending", "deps": ["install_deps"]},
    {"id": 31, "name": "dataloader_optimized", "group": "C", "status": "pending", "deps": ["tile_dataset_class"]},
    {"id": 32, "name": "class_weights", "group": "C", "status": "pending", "deps": ["tile_dataset_class"]},

    # GROUP D
    {"id": 33, "name": "model_efficientnet", "group": "D", "status": "pending", "deps": ["install_deps"]},
    {"id": 34, "name": "model_resnet50", "group": "D", "status": "pending", "deps": ["install_deps"]},
    {"id": 35, "name": "attention_pooling", "group": "D", "status": "pending", "deps": ["model_efficientnet"]},
    {"id": 36, "name": "loss_weighted_ce", "group": "D", "status": "pending", "deps": []},
    {"id": 37, "name": "loss_ordinal", "group": "D", "status": "pending", "deps": []},
    {"id": 38, "name": "optimizer_setup", "group": "D", "status": "pending", "deps": []},
    {"id": 39, "name": "amp_training", "group": "D", "status": "pending", "deps": ["install_torch_cuda"]},
    {"id": 40, "name": "cudnn_benchmark", "group": "D", "status": "pending", "deps": ["install_torch_cuda"]},
    {"id": 41, "name": "autotune_batch", "group": "D", "status": "pending", "deps": ["install_torch_cuda"]},
    {"id": 42, "name": "gpu_util_monitor", "group": "D", "status": "pending", "deps": ["verify_gpu"]},
    {"id": 43, "name": "train_loop", "group": "D", "status": "pending", "deps": ["model_efficientnet", "dataloader_optimized", "loss_weighted_ce", "optimizer_setup"]},
    {"id": 44, "name": "early_stopping", "group": "D", "status": "pending", "deps": ["train_loop"]},
    {"id": 45, "name": "resume_checkpoint", "group": "D", "status": "pending", "deps": ["train_loop"]},
    {"id": 46, "name": "ema_weights", "group": "D", "status": "pending", "deps": ["train_loop"]},
    {"id": 47, "name": "kfold_cv", "group": "D", "status": "pending", "deps": ["train_loop"]},
    {"id": 48, "name": "hyperparam_sweep", "group": "D", "status": "pending", "deps": ["train_loop"]},

    # GROUP E
    {"id": 49, "name": "metric_qwk", "group": "E", "status": "pending", "deps": []},
    {"id": 50, "name": "metric_confusion", "group": "E", "status": "pending", "deps": []},
    {"id": 51, "name": "metric_per_grade_f1", "group": "E", "status": "pending", "deps": []},
    {"id": 52, "name": "metric_roc_auc", "group": "E", "status": "pending", "deps": []},
    {"id": 53, "name": "slide_level_eval", "group": "E", "status": "pending", "deps": ["train_loop"]},
    {"id": 54, "name": "external_validation", "group": "E", "status": "pending", "deps": ["slide_level_eval", "download_tcga_prad"]},
    {"id": 55, "name": "error_analysis", "group": "E", "status": "pending", "deps": ["slide_level_eval"]},
    {"id": 56, "name": "calibration", "group": "E", "status": "pending", "deps": ["slide_level_eval"]},
    {"id": 57, "name": "bench_vs_pathologist", "group": "E", "status": "pending", "deps": ["slide_level_eval"]},
    {"id": 58, "name": "eval_report_writer", "group": "E", "status": "pending", "deps": ["slide_level_eval"]},

    # GROUP F
    {"id": 59, "name": "gradcam_core", "group": "F", "status": "pending", "deps": ["model_efficientnet"]},
    {"id": 60, "name": "gradcam_overlay", "group": "F", "status": "pending", "deps": ["gradcam_core"]},
    {"id": 61, "name": "gradcampp", "group": "F", "status": "pending", "deps": ["gradcam_core"]},
    {"id": 62, "name": "attention_maps", "group": "F", "status": "pending", "deps": ["attention_pooling"]},
    {"id": 63, "name": "tile_importance", "group": "F", "status": "pending", "deps": ["attention_pooling"]},
    {"id": 64, "name": "gradcam_explain_llm", "group": "F", "status": "pending", "deps": ["gradcam_overlay", "ollama_client"]},
    {"id": 65, "name": "saliency_sanity", "group": "F", "status": "pending", "deps": ["gradcam_core"]},
    {"id": 66, "name": "interpretability_gallery", "group": "F", "status": "pending", "deps": ["gradcam_overlay"]},

    # GROUP G
    {"id": 67, "name": "inference_api", "group": "G", "status": "pending", "deps": ["slide_level_eval"]},
    {"id": 68, "name": "batch_inference", "group": "G", "status": "pending", "deps": ["inference_api"]},
    {"id": 69, "name": "streamlit_demo", "group": "G", "status": "pending", "deps": ["inference_api", "gradcam_overlay"]},
    {"id": 70, "name": "model_export", "group": "G", "status": "pending", "deps": ["train_loop"]},
    {"id": 71, "name": "onnx_runtime_infer", "group": "G", "status": "pending", "deps": ["model_export"]},
    {"id": 72, "name": "dockerfile", "group": "G", "status": "pending", "deps": []},
    {"id": 73, "name": "remote_access", "group": "G", "status": "pending", "deps": ["streamlit_demo"]},
    {"id": 74, "name": "health_check", "group": "G", "status": "pending", "deps": []},

    # GROUP H
    {"id": 75, "name": "smoke_tests", "group": "H", "status": "pending", "deps": []},
    {"id": 76, "name": "unit_tests_data", "group": "H", "status": "pending", "deps": []},
    {"id": 77, "name": "unit_tests_metrics", "group": "H", "status": "pending", "deps": []},
    {"id": 78, "name": "ci_github_actions", "group": "H", "status": "pending", "deps": []},
    {"id": 79, "name": "pre_commit", "group": "H", "status": "pending", "deps": []},
    {"id": 80, "name": "config_validation", "group": "H", "status": "pending", "deps": []},
    {"id": 81, "name": "deploy_doctor", "group": "H", "status": "pending", "deps": []},
    {"id": 82, "name": "readme_writer", "group": "H", "status": "pending", "deps": []},
    {"id": 83, "name": "model_card", "group": "H", "status": "pending", "deps": []},
    {"id": 84, "name": "data_doc", "group": "H", "status": "pending", "deps": []},
    {"id": 85, "name": "results_doc", "group": "H", "status": "pending", "deps": []},
    {"id": 86, "name": "changelog", "group": "H", "status": "pending", "deps": []},
    {"id": 87, "name": "contributing_doc", "group": "H", "status": "pending", "deps": []},
    {"id": 88, "name": "secrets_guard", "group": "H", "status": "pending", "deps": []},

    # GROUP I
    {"id": 89, "name": "self_supervised_pretrain", "group": "I", "status": "pending", "deps": []},
    {"id": 90, "name": "mil_model", "group": "I", "status": "pending", "deps": []},
    {"id": 91, "name": "vit_backbone", "group": "I", "status": "pending", "deps": []},
    {"id": 92, "name": "foundation_features", "group": "I", "status": "pending", "deps": []},
    {"id": 93, "name": "tta", "group": "I", "status": "pending", "deps": []},
    {"id": 94, "name": "ensemble", "group": "I", "status": "pending", "deps": []},
    {"id": 95, "name": "active_learning", "group": "I", "status": "pending", "deps": []},
    {"id": 96, "name": "finetune_on_pes_data", "group": "I", "status": "pending", "deps": []},

    # GROUP J
    {"id": 97, "name": "skill_registry", "group": "J", "status": "pending", "deps": []},
    {"id": 98, "name": "next_skill_picker", "group": "J", "status": "pending", "deps": []},
    {"id": 99, "name": "loop_runner", "group": "J", "status": "pending", "deps": []},
    {"id": 100, "name": "progress_reporter", "group": "J", "status": "pending", "deps": []},
    {"id": 101, "name": "results_writer", "group": "K", "status": "pending", "deps": ["eval_kappa", "confusion_matrix"]},
    {"id": 102, "name": "figure_generator", "group": "K", "status": "pending", "deps": ["results_writer"]},
    {"id": 103, "name": "paper_scaffold", "group": "K", "status": "pending", "deps": ["figure_generator"]},
    {"id": 104, "name": "paper_autofill", "group": "K", "status": "pending", "deps": ["paper_scaffold"]}
]

os.makedirs('C:\\Users\\absh5\\.gemini\\antigravity\\scratch\\prostate-cadx\\skills', exist_ok=True)
registry_path = 'C:\\Users\\absh5\\.gemini\\antigravity\\scratch\\prostate-cadx\\skills\\registry.json'
with open(registry_path, 'w') as f:
    json.dump(skills, f, indent=2)

print("skills/registry.json initialized successfully.")
