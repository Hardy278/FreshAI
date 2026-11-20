from src.models.trainer import run_training_pipeline


if __name__ == "__main__":
    results = run_training_pipeline()

    print("\n===== 实验完成 =====")
    print("最佳迭代轮数：", results["best_iteration"])
    print(results["comparison"])
