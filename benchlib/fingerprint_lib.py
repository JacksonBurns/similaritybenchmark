from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator

N_BITS = 2048

_MORGAN_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)
_RDK_FPGEN = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=N_BITS)

# dictionary
FP_LOOKUP = dict(
    morgan=lambda mol: _MORGAN_FPGEN.GetFingerprint(mol),
    rdk=lambda mol: _RDK_FPGEN.GetFingerprint(mol),
)
