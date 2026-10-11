import tarfile, glob, os

path = glob.glob("data/**/*.tar.xz", recursive=True)[0]
print("Found:", path)

out = "data/extracted"
os.makedirs(out, exist_ok=True)

with tarfile.open(path, "r:xz") as tar:
    members = tar.getnames()
    print("Files inside:", len(members))
    print("First 20:")
    for m in members[:20]:
        print("  ", m)
    tar.extractall(out)

print("\nExtracted to:", out)