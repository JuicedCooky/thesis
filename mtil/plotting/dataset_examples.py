"""
Show one example image from each MTIL dataset with its class label.
Saves the figure to dataset_examples.png next to this script.
"""
import os
import sys
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; no display needed
import matplotlib.pyplot as plt
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_PATH  = os.path.join(os.path.dirname(__file__), "dataset_examples.png")

# ── custom classnames (datasets whose torchvision labels need mapping) ──────
FLOWERS_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]

EUROSAT_CLASSES = [
    "annual crop land", "forest", "brushland or shrubland", "highway or road",
    "industrial buildings", "pasture land", "permanent crop land",
    "residential buildings", "river", "lake or sea",
]


# ── helpers ─────────────────────────────────────────────────────────────────
def classname(ds, label, custom=None):
    """Map integer label to a human-readable class name."""
    if custom is not None:
        return custom[label]
    if hasattr(ds, "classes"):
        raw = ds.classes[label]
    elif hasattr(ds, "categories"):
        raw = ds.categories[label]
    else:
        return f"class {label}"
    # SUN397 stores paths like /a/abbey
    if raw.startswith("/"):
        raw = raw.rstrip("/").split("/")[-1]
    return raw.replace("_", " ")


def try_sample(name, build_fn, custom_classes=None):
    """
    Call build_fn() to get a torchvision dataset, then return
    a random (PIL Image, classname string), or None on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = build_fn()
        idx = random.randrange(len(ds))
        img, label = ds[idx]
        if not isinstance(img, Image.Image):
            # e.g. already a numpy array
            img = Image.fromarray(np.uint8(img))
        # Ensure RGB for uniform display
        img = img.convert("RGB")
        cn = classname(ds, label, custom_classes)
        return img, cn
    except Exception as e:
        print(f"  [{name}] skipped - {type(e).__name__}: {e}")
        return None


# ── dataset loaders ──────────────────────────────────────────────────────────
def make_datasets():
    """
    Returns a list of (display_name, loader_fn, optional_custom_classes).
    Each loader returns a torchvision-style dataset with transform=None.
    """
    from torchvision import datasets as tvd

    loc = DATA_DIR

    return [
        ("MNIST", lambda: tvd.MNIST(loc, train=False, download=True),
         None),

        ("CIFAR-10", lambda: tvd.CIFAR10(loc, train=False, download=True),
         None),

        ("CIFAR-100", lambda: tvd.CIFAR100(loc, train=False, download=True),
         None),

        ("DTD", lambda: tvd.DTD(loc, split="test", download=True),
         None),

        ("EuroSAT", lambda: tvd.EuroSAT(loc, download=True),
         EUROSAT_CLASSES),

        ("Flowers102", lambda: tvd.Flowers102(loc, split="test", download=True),
         FLOWERS_CLASSES),

        ("Food-101", lambda: tvd.Food101(loc, split="test", download=True),
         None),

        ("Oxford Pets", lambda: tvd.OxfordIIITPet(loc, split="test", download=True),
         None),

        ("Aircraft", lambda: tvd.FGVCAircraft(loc, split="test", download=True),
         None),

        ("Caltech-101", lambda: tvd.Caltech101(loc, download=True),
         None),

        ("SUN397", lambda: tvd.SUN397(loc, download=False),
         None),

        ("Stanford Cars", lambda: tvd.StanfordCars(loc, split="test", download=False),
         None),
    ]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Data directory: {DATA_DIR}")
    entries = make_datasets()

    samples = []
    for name, loader, custom in entries:
        result = try_sample(name, loader, custom)
        if result is not None:
            samples.append((name, *result))

    if not samples:
        print("No datasets loaded — nothing to show.")
        sys.exit(1)

    n = len(samples)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 4.0))
    axes = np.array(axes).reshape(-1)  # flatten for uniform indexing

    for i, (ds_name, img, cn) in enumerate(samples):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(
            f"Dataset: {ds_name}\nLabel: {cn}",
            fontsize=8, pad=4, loc="center",
            fontfamily="monospace",
        )
        ax.axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Dataset Examples with Class Labels", fontsize=12, fontweight="bold")
    plt.tight_layout(h_pad=3.0)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
