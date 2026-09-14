"""Section 8.5, step 2: prevent direct concept cueing and construction leakage."""


def check_concept_separation(example, direction, complex_data):
    if direction["direction_family"] != "concept":
        return
    injected = direction["concept"]
    if example["probed_concept"].casefold() == injected.casefold():
        raise ValueError("The probed and injected concepts must differ")
    # The historical complex-vector extractor uses every positive and negative
    # example. Never evaluate a sentence used to construct this injected direction.
    if direction.get("concept_dataset") == "complex_data":
        if injected not in complex_data:
            raise ValueError("Cannot verify construction provenance for this complex concept")
        construction = [s for group in complex_data[injected] for s in group]
        normalize = lambda s: " ".join(s.casefold().split())
        if normalize(example["sentence"]) in set(map(normalize, construction)):
            raise ValueError("Evaluation sentence occurs in the injected direction's construction data")
