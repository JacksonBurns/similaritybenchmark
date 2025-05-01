from pathlib import Path
from functools import partial

from rdkit import Chem, DataStructs
import psutil
from tqdm.contrib.concurrent import process_map
from scipy.stats import spearmanr
import numpy as np

from benchlib.chembl import SMILES_LOOKUP
from benchlib.fingerprint_lib import FP_LOOKUP


def read_benchmark(fname):
    """Returns list of SMILES for each group in a benchmark repetition.

    i.e. read_benchmark('0.txt') -> [['C', 'CC', ...], ...]

    Parameters
    ----------
    fname : str
        Input file containing lines of ChEMBL IDs separate by whitespace

    Returns
    ------
    list[list[str]]
        Lists of lists of SMILES
    """
    with open(fname) as file:
        return [[SMILES_LOOKUP[id] for id in line.rstrip().split()] for line in file.readlines()]


def get_rdkitmols(dataset: list[list[str]]) -> list[list[Chem.Mol]]:
    """Converts list[list[str]] into corresponding rdkit mols

    Parameters
    ----------
    dataset : list[list[str]]
        List of lists of SMILES strings

    Returns
    -------
    list[list[Chem.Mol]]
        RDKit Molecules with fragment-containing molecules defined by their largest fragment
    """
    out_list = []
    for group in dataset:
        tmp = []
        for smi in group:
            mol = Chem.MolFromSmiles(smi)
            if "." in smi:
                frags = list(Chem.GetMolFrags(mol, asMols=True))
                frags.sort(key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
                mol = frags[0]
            tmp.append(mol)
        out_list.append(tmp)
    return out_list


def evaluate_similarity_method(dataset, results_dir):
    """Checks the similarity between each group of molecules and its reference, writing to a file"""
    # Setup results dir
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    reference_order = [4, 3, 2, 1]
    for fp_name, fp_calculator in FP_LOOKUP.items():
        with open(results_dir / (fp_name + ".txt"), "a") as file:
            for mol_group in get_rdkitmols(dataset):
                ref_fp = fp_calculator(mol_group[0])
                similarities = []
                for other_mol in mol_group[1:]:
                    other_fp = fp_calculator(other_mol)
                    if fp_name in {"morgan", "rdk"}:
                        sim = DataStructs.FingerprintSimilarity(ref_fp, other_fp)
                    else:
                        raise NotImplementedError()
                    similarities.append(sim)
                # calculate correlation
                corr = spearmanr(reference_order, similarities).statistic
                if np.isnan(corr):
                    corr = -1
                file.write(f"{corr}\n")


# helper function for parallelism
def _process_repetition(repetition, benchmark, outdir):
    filename = Path(f"{benchmark}/{benchmark}/dataset/{repetition}.txt")
    smiles_groups = read_benchmark(filename)
    evaluate_similarity_method(
        smiles_groups,
        outdir / str(repetition),
    )


if __name__ == "__main__":
    for benchmark in ["SingleAssay", "MultiAssay"]:
        outdir = Path(f"{benchmark}/results")
        outdir.mkdir()  # raises error if already present
        repetitions = range(4)  # Should be range(1_000) for full run
        num_processes = psutil.cpu_count(logical=False)
        process_func = partial(_process_repetition, benchmark=benchmark, outdir=outdir)
        process_map(
            process_func,
            repetitions,
            max_workers=num_processes,
            desc=benchmark,
            chunksize=1,
        )
