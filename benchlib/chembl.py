from pathlib import Path

with open(Path(__file__).parent / "chembl_20.smi") as file:
    SMILES_LOOKUP = dict(
        (chembl_id, smiles) for (smiles, chembl_id) in filter(lambda split_line: len(split_line) == 2, map(str.split, file.readlines()))
    )
