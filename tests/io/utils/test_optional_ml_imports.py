"""Core annotation operations do not initialize an installed Torch runtime."""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_io_annotations_and_stereochemistry_do_not_import_torch():
    source = str(Path(__file__).resolve().parents[3] / "src")
    environment = dict(os.environ, PYTHONPATH=source + os.pathsep + os.environ.get("PYTHONPATH", ""))
    script = dedent("""
        import sys, numpy as np
        assert 'torch' not in sys.modules
        import atomworks
        from atomworks.io.utils.annotator import ensure_annotations
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
        from atomworks.ml.transforms.rdkit_utils import get_stereochemistry
        from atomworks.ml.transforms.base import Compose,Identity
        from biotite.structure import AtomArray
        from rdkit import Chem
        assert 'torch' not in sys.modules
        array=AtomArray(4)
        array.res_name[:]='ALA';array.chain_id[:]='A';array.res_id[:]=[2,2,4,4]
        array.atom_name[:]=['N','CA','N','CA'];array.element[:]=['N','C','N','C']
        ensure_annotations(array,'within_chain_res_idx','within_chain_source_res_idx')
        np.testing.assert_array_equal(array.within_chain_res_idx,[0,0,1,1])
        np.testing.assert_array_equal(array.within_chain_source_res_idx,[0,0,2,2])
        result=get_stereochemistry(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))
        assert len(result['chiral_centers'])==1
        assert Compose([Identity()],track_rng_state=False)({'value':3})['value']==3
        assert 'torch' not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", script], env=environment, check=True, capture_output=True, text=True)
