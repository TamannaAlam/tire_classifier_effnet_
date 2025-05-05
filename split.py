import pathlib, random, shutil

root = pathlib.Path("tires_images")
train_root, val_root = root.parent / "train", root.parent / "val"

for split_root in (train_root, val_root):
    (split_root / "normal").mkdir(parents=True, exist_ok=True)
    (split_root / "winter").mkdir(parents=True, exist_ok=True)

for cls in ["normal", "winter"]:
    # grab every file in all sub-folders
    imgs = [p for p in (root/cls).rglob("*") if p.is_file()]
    random.shuffle(imgs)
    val_n = int(0.2 * len(imgs))

    for i, src in enumerate(imgs):
        dst_root = val_root/cls if i < val_n else train_root/cls
        dst_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_root/src.name)
