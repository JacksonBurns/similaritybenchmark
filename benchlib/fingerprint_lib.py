from pathlib import Path

from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Mol
from rdkit.Avalon import pyAvalonTools as fpAvalon
import torch

from chemprop.data import BatchMolGraph
from chemprop.nn.agg import MeanAggregation
from chemprop.nn import RegressionFFN
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN

N_BITS = 2048

# classical methods
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)
_RDK = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=N_BITS)
_AP = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
_TT = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS)

"""
# deep methods
_CHEMELEON = MPNN(
    message_passing=torch.load(Path(__file__).parent / "chemeleon_mp.pt", map_location="cpu", weights_only=False),
    agg=MeanAggregation(),  # matches original chemeleon training
    predictor=RegressionFFN(),  # not actually used
)
_CHEMELEON.eval()
_CHEMELEON.to(device="cuda:0")
_CHEMELON_FEAT = SimpleMoleculeMolGraphFeaturizer()
"""

@torch.no_grad
def chemeleon_fingerprint(mol: Mol):
    bmg = BatchMolGraph([_CHEMELON_FEAT(mol)])
    bmg.to(device="cuda:0")
    return _CHEMELEON.fingerprint(bmg).numpy(force=True).flatten()

FP_LOOKUP = dict(
    # chemeleon=chemeleon_fingerprint,
    morgan=lambda mol: _MORGAN.GetFingerprint(mol),
    rdk=lambda mol: _RDK.GetFingerprint(mol),
    maccs=lambda mol: MACCSkeys.GenMACCSKeys(mol),
    atom_pair=lambda mol: _AP.GetFingerprint(mol),
    top_tor=lambda mol: _TT.GetFingerprint(mol),
    avalon=lambda mol: fpAvalon.GetAvalonFP(mol, N_BITS),
)
