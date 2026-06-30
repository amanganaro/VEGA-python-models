from rdkit import Chem
from rdkit.Chem import Descriptors

smiles_list = [
    "O=C(O)C(NC(C(=O)N2CC(NC(=O)CC(N)Cc1cc(F)c(F)cc1(F))CC2(C(=O)O))CCCC)CCc3ccccc3",
    "O=C(O)CCC(NC(=O)C(NC(=O)C(Cc1onc(c1)c2ccc(cc2)c3cccc(c3)Cl)CP(=O)(O)c4ccc(cc4)Br)CCC(=O)O)C(=O)N",
    "O=C(O)CCC(N)C(=O)NC(C(=O)N4CCCC4(C(=O)NC(C(=O)N3CCCC3(C(=O)NC(C(=O)NC(C(=O)N2CCCC2(C(=O)N1CCCC1(C(=O)O)))C(C)CC)CCC(=O)N))CCCN=C(N)N))Cc6c[nH]c5ccccc56",
    "O=C(O)CCC(N)C(=O)NC(C(=O)NC(C(=O)NC(C(=O)N1CCCC1(C(=O)O))C)Cc3c[nH]c2ccccc23)CCCCN",
    "O=C(O)c6ccccc6(c1ccc(cc1)Cn2c3cc(cc(c3(nc2CCC))C)c5nc4ccccc4n5C)",
    "O=C(O)C(NC(=O)C(CC2ON=C(c1ccccc1)C2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)C",
    "O=C(O)C(NC(=O)C(CC2ON=C(c1ccc(cc1)C(F)(F)F)C2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)Cc6c[nH]c5ccccc56",
    "O=C(O)C4N(C(=O)C(NC(=O)C(NC(=O)C(NC(=O)C1NC(=O)CC1)CCCCN)Cc3c[nH]c2ccccc23)C)CCC4",
    "O=C(O)C(NC(=O)C(CC2ON=C(c1ccccc1)C2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)Cc6c[nH]c5ccccc56",
    "O=C(O)C(NC(=O)C1(CCCC1)CP(=O)(O)C(NC(=O)C(NS(=O)(=O)C)CCCCN)Cc2ccccc2)Cc4c[nH]c3ccccc34",
    "O=C(OCc1ccccc1)NC(Cc2ccccc2)P(=O)(O)CC(C(=O)NC(C(=O)O)Cc4c[nH]c3ccccc34)Cc5ccc(cc5)c6ccccc6",
    "O=C(OCc1ccccc1)C5N(C(=O)CCC(=O)C(NC(=O)c2ccccc2)Cc4ccc(OCc3ccccc3)cc4)CCC5",
    "O=C(O)C(NC(=O)C(CC3ON=C(c1ccc(cc1)c2ccccc2)C3)CP(=O)(O)C(NC(=O)OCc4ccccc4)Cc5ccccc5)Cc7c[nH]c6ccccc67",
    "O=C(O)C7N(C(=O)C6N(C(=O)C(NC(=O)C(NC(=O)C5N(C(=O)C(NC(=O)C4N(C(=O)C(NC(=O)C1NC(=O)CC1)Cc3c[nH]c2ccccc23)CCC4)CCCNC(=N)N)CCC5)CCC(=O)N)C(C)CC)CCC6)CCC7",
    "O=C(O)C(NC(=O)C(NC(=O)N1CCCC1C(NC(C(=O)N2CCCC2(C(=O)O))C)C(=O)O)Cc3ccccc3)Cc4ccccc4",
    "O=C(OC)C2N(C(=O)C(N(C(=O)C(NC(=O)C(NC(=O)CC(O)C(NC(=O)C(NC(=O)C(NC(=O)C(OC(=O)C(OC(=O)C(N(C)C)C(C)C)C(C)C)C(C)C)CC(C)C)CC(=O)N)CC(C)C)C(O)C)C)C)Cc1ccccc1)CCC2",
    "O=C(O)C(NC(C(=O)N2CC(NC(=O)CC(N)Cc1cc(F)c(F)cc1(F))CC2(C(=O)O))CCC)CCc3ccccc3",
    "O=C(OC9C%11C%12(OC(=O)C)(COC%12(CC(O)C%11(C(=O)C(OC(=O)C)C%10=C(C)C(OC(=O)C(OC(=O)CCC(=O)NC(C(=O)NC(C(=O)N4CCCC4(C(=O)NC(C(=O)N3CCCC3(C(=O)NC(C(=O)NC(C(=O)N2CCCC2(C(=O)N1CCCC1(C(=O)N)))C(C)CC)CCC(=O)N))CCCNC(=N)N))Cc6c[nH]c5ccccc56)CCC(=O)O)C(NC(=O)c7ccccc7)c8ccccc8)CC9(O)C%10(C)C)(C))))c%13ccccc%13",
    "O=C(O)C(NC(=O)C(Cc1onc(c1)c2ccccc2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)Cc5ccc(O)cc5",
    "O=C(O)C(NC(=O)C(Cc1onc(c1)c2ccccc2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)Cc5ccccc5",
    "O=C(O)C(NC(=O)C(Cc1onc(c1)c2ccccc2)CP(=O)(O)C(NC(=O)OCc3ccccc3)Cc4ccccc4)Cc6c[nH]c5ccccc56",
    "O=C(O)C6N5C(=O)C(NC(=O)C(Cc1ccccc1)SSC(C(=O)NC3C(=O)N2C(C(=O)O)CCCC2SCC3)Cc4ccccc4)CCSC5CCC6"
]

def test_molecule_param_for_cddd_preprocess():
    for i, sml in enumerate(smiles_list):

        print("(smiles n:" , i+1,") SMILES: ", sml)
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        print("LogP: ", logp)
        print("Molecular weight: ", mol_weight)
        print("Number of heavy atoms: ", num_heavy_atoms)
        if not((logp > -5) & (logp < 7)):
            print("The molecule exceed in LogP (it must be -5 < logp < 7)")
        if not((mol_weight > 12) & (mol_weight < 600)):
            print("The molecule exceed in molecular weight, (it must be 12 < molecular weight < 600)")
        if not((num_heavy_atoms > 3) & (num_heavy_atoms < 50)):
            print("The molecule exceed in number of heavy atoms, (it must be 3 < num heavy atoms < 50)")
        print("\n")

if __name__ == '__main__':
    test_molecule_param_for_cddd_preprocess()