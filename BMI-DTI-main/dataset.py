import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))
atom_list = []
valence = []
hydrogens = []

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items(): #遍历
            s = sorted(list(s)) #按升序排列
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))  #没看懂
            self.dim += len(s)
        #根据allowable_set创建features_mapping  （具体怎么做到的）
    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs) #获取每个特征的取值
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        if atom.GetSymbol() not in atom_list:
            atom_list.append(atom.GetSymbol())
        return atom.GetSymbol()

    def n_valence(self, atom):
        if atom.GetTotalValence() not in valence:
            valence.append(atom.GetTotalValence())
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        if atom.GetTotalNumHs()not in valence:
            hydrogens.append(atom.GetTotalNumHs())
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()



class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None: #即没有化学键的情况 这个判断不会生效的
            output[-1] = 1.0
            return output
        output = super().encode(bond) #调用父类的encode
        return output

    def bond_type(self, bond):#返回化学键的类型
        return bond.GetBondType().name.lower()

    def conjugated(self, bond): #返回化学键是否为共扼键
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol":  {'Ca', 'Na','C', 'O', 'N', 'H', 'B', 'S', 'V', 'P', 'F', 'Cl', 'Br', 'Al', 'As', 'Hg', 'I', 'Co', 'Fe', 'Se', 'Cu', 'W', 'Ru', 'Zn', 'Si', 'Mo', 'Au', 'Sn', 'Te'},
        "n_valence": {0, 1, 2, 3, 4, 5, 6}, #总化合价
        "n_hydrogens": {0, 1, 2, 3, 4},#氢原子个数
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)



def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds(): #没有键就会跳
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms())) #创建一个包含分子中所有原子索引打集合
    isolate_atoms = atoms_set - set(i_jbond_dict.keys()) #找到没有键连接打原子索引  keys() 方法返回字典中的所有键
    bond = bond_dict['nan'] #孤立打原子设置默认的值
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))  #这个键指向自己

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius):#生成原子特征的函数，
    ##通过组合原子打特征和邻居的特征 捕捉原子之间的关系和依赖
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


# mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
# if mol.GetNumAtoms() == 0:
#     print("Failed to create molecule")
# if mol is None:
#     print("Failed to create molecule")
# atoms = create_atoms(mol)
# i_jbond_dict = create_ijbonddict(mol)
# #radius 用来限制从每个原子开始遍历邻居原子的层数，越大越能包含邻居信息
# compounds.append(atom_features(atoms, i_jbond_dict, radius))
# embeddings = []
# for atom in mol.GetAtoms():
#     embeddings.append(atom_featurizer.encode(atom)) #每个原子的特征向量编码
# node_embeddings.append(embeddings)
# edges,edge_attrr=[],[]
# for bond in mol.GetBonds():
#     edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
#     edge_attrr.append(bond_featurizer.encode(bond))
#     edge_attrr.append(bond_featurizer.encode(bond))
# if len(edges)==0:
#     # atoms_set = set(range(mol.GetNumAtoms())) #创建一个包含分子中所有原子索引打集合
#     # isolate_atoms = atoms_set - set(i_jbond_dict.keys()) #找到没有键连接打原子索引  keys() 方法返回字典中的所有键
#     # bond = bond_dict['nan'] #孤立打原子设置默认的值
#     # for a in isolate_atoms:
#     #     i_jbond_dict[a].append((a, bond))  #这个键指向自己
#     bond = None
#     edges.append([mol.GetNumAtoms(), mol.GetNumAtoms()])
#     edges.append([mol.GetNumAtoms(), mol.GetNumAtoms()])
#     edge_attrr.append(bond_featurizer.encode(bond))
#     edge_attrr.append(bond_featurizer.encode(bond))
def atom_edge(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)
    #radius 用来限制从每个原子开始遍历邻居原子的层数，越大越能包含邻居信息
    atom_feature = atom_features(atoms, i_jbond_dict, 2)
    edges,edge_attrr=[],[]
    embeddings = []
    for atom in mol.GetAtoms():
        embeddings.append(atom_featurizer.encode(atom)) #每个原子的特征向量编码
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attrr.append(bond_featurizer.encode(bond))
        edge_attrr.append(bond_featurizer.encode(bond))
    if len(edges)==0:
        atoms_set = set(range(mol.GetNumAtoms())) #创建一个包含分子中所有原子索引打集合
        isolate_atoms = atoms_set - set(i_jbond_dict.keys()) #找到没有键连接打原子索引  keys() 方法返回字典中的所有键
        bond = bond_dict['nan'] #孤立打原子设置默认的值
        for a in isolate_atoms:
            i_jbond_dict[a].append((a, bond))  #这个键指向自己
        bond = None
        edges.append([mol.GetNumAtoms(), mol.GetNumAtoms()])
        edges.append([mol.GetNumAtoms(), mol.GetNumAtoms()])
        edge_attrr.append(bond_featurizer.encode(bond))
        edge_attrr.append(bond_featurizer.encode(bond))
    embeddings = np.array(embeddings)
    embeddings = torch.tensor(embeddings)
    atom_feature = torch.tensor(atom_feature)
    edges = torch.tensor(edges).T
    #edges = torch.tensor(edges).T
    edge_attrr = np.array(edge_attrr, dtype=np.float32)
    edge_attrr = torch.tensor(edge_attrr)
    return atom_feature,embeddings,edges,edge_attrr

def edge_pad(str):
    rows_list = [tensor.size(0) for tensor in str]
    cols_list = [tensor.size(1) for tensor in str]
    max_rows = max(rows_list)
    max_cols = max(cols_list)
    mask = torch.zeros((len(str), max_rows, max_cols))
    # 创建三维张量并填充数据和掩码
    tensor_combined = torch.zeros((len(str), max_rows, max_cols))
    for i, tensor in enumerate(str):
        rows, cols = tensor.size()
        tensor_combined[i, :rows, :cols] = tensor
        mask[i, :rows, :cols] = 1
    return tensor_combined,mask

def atom_pad(str):
    rows_list = [tensor.size(0) for tensor in str]

    max_rows = max(rows_list)

    mask = torch.zeros((len(str), max_rows))
    # 创建三维张量并填充数据和掩码
    tensor_combined = torch.zeros((len(str), max_rows))
    for i, tensor in enumerate(str):
        rows = tensor.size(0)
        tensor_combined[i, :rows] = tensor
        mask[i, :rows] = 1
    new_tensor = tensor_combined*mask
    return new_tensor

def collate_fn(batch_data,max_d=100,max_p=1200):
    N = len(batch_data)
    # compound_new = torch.zeros((N, max_d), dtype=torch.long)
    # protein_new = torch.zeros((N, max_p), dtype=torch.long)
    # labels_new = torch.zeros(N, dtype=torch.float)

    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    compound_mask = torch.zeros((N, compound_max))
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    protein_mask = torch.zeros((N, protein_max))
    labels_new = torch.zeros(N, dtype=torch.long)

    atoms,node,edge_index,edge_attr = [],[],[],[]


    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()
        compoundstr, proteinstr, label = pair[-3], pair[-2], pair[-1]
        smiles_len = len(compoundstr)
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET,compound_max))
        compound_new[i] = compoundint
        if smiles_len > compound_max:
            compound_mask[i,:] = 1
        else:
            compound_mask[i,:smiles_len] = 1

        atom,embeding,edge,attr = atom_edge(compoundstr)
        atoms.append(atom)
        node.append(embeding)
        edge_index.append(edge)
        edge_attr.append(attr)

        pro_len = len(proteinstr)
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET,protein_max))
        protein_new[i] = proteinint
        if pro_len > protein_max:
            protein_mask[i, :] = 1
        else:
            protein_mask[i, :pro_len] = 1

        labels_new[i] = int(float(label))


        # labels_new[i] = np.float(label)


    atoms = atom_pad(atoms)
    node,node_mask = edge_pad(node)
    edge_index,edge_mask = edge_pad(edge_index)
    edge_index = edge_index.long()
    edge_attr,mask = edge_pad(edge_attr)
    #edges_index = torch.stack(edge_index,dim=0)
    return (compound_new,node_mask,node,edge_index,edge_attr, protein_new,compound_mask, protein_mask, labels_new)

