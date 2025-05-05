from pathlib import Path
from functools import partial
import os

from rdkit import Chem, DataStructs
import psutil
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
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
                    if fp_name in {"morgan", "rdk", "avalon", "maccs"}:
                        sim = DataStructs.FingerprintSimilarity(ref_fp, other_fp)
                    elif fp_name in {"atom_pair", "top_tor"}:
                        sim = DataStructs.DiceSimilarity(ref_fp, other_fp)
                    elif fp_name in {"minimol", "chemeleon"}:
                        # scipy implements cosine as the DISTANCE, we convert back to SIMILARITY
                        sim = 1 - cosine(ref_fp, other_fp)
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
    # to keep comparisons balanced between repetitions balanced between all repetitions, we cut each
    # one off at the size of the smallest one. This number is known a-priori based on the data and
    # is thus hard coded here (I found it by running the code _without_ this line *dumbass* and then
    # using BASH commands to check which result file was the smallest)
    smiles_groups = smiles_groups[:3629] if benchmark == "MultiAssay" else smiles_groups[:4563]
    evaluate_similarity_method(
        smiles_groups,
        outdir / str(repetition),
    )


if __name__ == "__main__":
    # can't use CUDA with default fork multiprocessing method
    for benchmark in ["SingleAssay", "MultiAssay"]:
        outdir = Path(f"{benchmark}/results")
        outdir.mkdir()  # raises error if already present
        repetitions = range(100)  # original study did 1k
        multi = os.getenv("ENABLE_PARALLEL", False)
        if multi:
            num_processes = psutil.cpu_count(logical=False)
            process_func = partial(_process_repetition, benchmark=benchmark, outdir=outdir)
            process_map(
                process_func,
                repetitions,
                max_workers=num_processes,
                desc=benchmark,
                chunksize=1,
            )
        else:
            for rep in tqdm(repetitions, desc=benchmark):
                _process_repetition(
                    rep,
                    benchmark,
                    outdir,
                )
