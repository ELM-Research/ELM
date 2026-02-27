from configs.config import get_args
from utils.dir_file_manager import DirFileManager

DIR_FILE_MANAGER = DirFileManager()

def extract_metrics(json_file):
    acc_mean = json_file["ACC"]["mean"]
    acc_std = json_file["ACC"]["std"]
    f1_mean = json_file["ACC"]["mean"]
    f1_std = json_file["ACC"]["std"]
    if "per_class_acc" in json_file:
        for per_class in json_file["per_class_acc"].keys():
            per_class_acc_mean = json_file["per_class_acc"][per_class]["mean"]
            per_class_acc_std = json_file["per_class_acc"][per_class]["std"]


def main():
    mode = "analyze"
    args = get_args(mode)

    for json_path in args.json_paths:
        json_file = DIR_FILE_MANAGER.open_json(json_path)

if __name__ == "__main__":
    main()