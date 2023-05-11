import os

def get_path(path, mode="train"):
    class_names = sorted(os.listdir(path))
    with open(f"Imagenet_{mode}_label.txt", "w") as f:
        for i in range(len(class_names)):
            base_path = os.path.join(path, class_names[i])
            file_names = sorted(os.listdir(base_path))
            for j in range(len(file_names)):
                f.write(f"{os.path.join(base_path, file_names[j])},{i}\n")
            

get_path("/lab/tmpig23c/u/andy/ILSVRC/Data/CLS-LOC/train/")
get_path("/lab/xingrui/distillation_cnn/data/ILSVRC/Data/CLS-LOC/validation", "val")