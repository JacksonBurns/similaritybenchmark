from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools as fpAvalon

N_BITS = 2048

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)
_RDK = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=N_BITS)
_AP = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
_TT = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS)

FP_LOOKUP = dict(
    morgan=lambda mol: _MORGAN.GetFingerprint(mol),
    rdk=lambda mol: _RDK.GetFingerprint(mol),
    maccs=lambda mol: MACCSkeys.GenMACCSKeys(mol),
    atom_pair=lambda mol: _AP.GetFingerprint(mol),
    top_tor=lambda mol: _TT.GetFingerprint(mol),
    avalon=lambda mol: fpAvalon.GetAvalonFP(mol, N_BITS),
)
