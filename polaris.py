#!/usr/bin/env python

"""
See
https://cobalt.itc.univie.ac.at/molcasforum/viewtopic.php?id=314
https://www.researchgate.net/post/Can_anyone_explain_how_I_can_calculate_the_polarizability_from_the_molcas_CASSCF_output_calculation
Computational Aspects of Electric Polarizability Calculations p. 255 ff.
"""

import itertools as it
import re
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

from jinja2 import Template
import numpy as np


np.set_printoptions(suppress=True, precision=4)


def prepare_input(calc_params, strength):
    def str2tpl(tpl_str):
        return Template(textwrap.dedent(tpl_str))

    scf_tpl_str = """
    &scf
     prorbitals
      0
    """
    scf_tpl = str2tpl(scf_tpl_str)

    ras_tpl_str = """
    &rasscf
     charge
      {{ calc.charge }}
     spin
      {{ calc.spin }}
     fileorb
      {{ calc.fileorb }}
     ciroot
      {{ calc.ciroot }} {{ calc.ciroot }} 1
     thrs
      1.0e-12,1.0e-4,1.0e-4
    """
    ras_tpl = str2tpl(ras_tpl_str)

    method_dict = {
        "scf": scf_tpl,
        "ras": ras_tpl,
    }

    # Render method string
    method_str = method_dict[calc_params["method"]].render(calc=calc_params)

    inp_tpl_str = """
    &gateway
     coord
      {{ calc.xyz }}
     basis
      {{ calc.basis }}
     ricd
     group
      nosym

    &seward

    &ffpt
     dipo
      {{ direction }} {{ strength }}

    {{ method_str }}
    """
    inp_tpl = str2tpl(inp_tpl_str)

    directions = "X Y Z".split()
    strengths = strength * np.array((1, -1))
    product = list(it.product(strengths, directions))

    job_input = "\n".join(
        [inp_tpl.render(direction=d, strength=s, calc=calc_params,
                        method_str=method_str)
         for s, d in product]
    )
    job_order = product
    return job_input, job_order


def run_molcas(job_input):
    assert "MOLCAS" in os.environ, "$MOLCAS isn't set!"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        job_fn  = "openmolcas.in"
        job_path = tmp_path / job_fn
        with open(job_path, "w") as handle:
            handle.write(job_input)

        out_fn = "openmolcas.out"
        out_path = tmp_path / out_fn
        args = f"pymolcas {job_fn}".split()
        with open(out_path, "w") as handle:
            result = subprocess.Popen(args, cwd=tmp_path,
                                      stdout=handle, stderr=subprocess.PIPE)
            result.wait()
        with open(out_path) as handle:
            text = handle.read()

        return text


def parse_log(text):
    root_re = "RASSCF root number\s*\d\s*Total energy:\s*([\d\.\-]+)"
    ens = np.array(re.findall(root_re, text), dtype=float)
    dpm_re = "X=\s*([\d\.E\-\+]+)\s*Y=\s*([\d\.E\-\+]+)\s*Z=\s*" \
             "([\d\.E\-\+]+)\s*Total=\s*([\d\.E\-\+]+)\s*"
    dpms = np.array(re.findall(dpm_re, text), dtype=float)
    return ens, dpms


def two_fields(diff1, diff2, strength):
    return ((2/3)*diff1 - (1/12)*diff2) / strength


def get_diff(calc_params, F):
    print(f"Running calculations for F={F}")
    job_input, job_order = prepare_input(calc_params, F)
    job_order_str = " ".join([f"({s:.3f} {d})" for s, d in job_order])
    text = run_molcas(job_input)
    print(job_order_str)

    fn_base = f"calc_{F:.4f}"
    log_fn = fn_base + ".log"
    dpms_fn = fn_base + "_dpms.dat"
    with open(log_fn, "w") as handle:
        handle.write(text)
    _, dpms = parse_log(text)
    np.savetxt(dpms_fn, dpms)

    # Drop the total component
    dpms = dpms[:,:3]
    # Reshape dpms into two halves, the plus-F half and the minus-F half
    dpms = dpms.reshape(2, -1, 3)
    # Convert from Debye to a.u.
    dpms /= 2.5418

    diff = dpms[0] - dpms[1]
    return diff


def run():
    fields = 2
    F0 = 0.002
    base = 2
    # Geometric series
    strengths = F0 * np.power(base, range(fields))

    nh3_ras_params = {
        "xyz": "/scratch/molcas_jobs/nh3_inversion/backup/01_relaxed_scan/nh3_inversion.Opt.15.xyz",
        "basis": "ano-rcc-vdzp",
        "charge": 0,
        "spin": 1,
        "fileorb": "/scratch/molcas_jobs/nh3_inversion/backup/05_casscf_pes/nh3_inversion.15.RasOrb",
        "ciroot": 2,
        "method": "ras",
    }

    form_hf_params = {
        "xyz": "/scratch/polarisierbarkeit/geometrien/formaldehyd.xyz",
        "basis": "aug-cc-pvdz",
        "charge": 0,
        "spin": 1,
        "method": "scf",
    }

    calc_params = form_hf_params
    # calc_params = nh3_ras_params

    diffs = [get_diff(calc_params, F) for F in strengths]
    print("Diffs")
    print(diffs)

    ff_funcs = {
        2: two_fields,
    }

    alphas = ff_funcs[fields](*diffs, F0)

    ax = "xyz"
    for i in range(3):
        a = alphas[i][i]
        print(f"α_{ax[i]}{ax[i]}: {a:.2f}")
    mean_alpha = np.sum(np.diag(alphas))/3
    print(f"mean(α) = {mean_alpha:.2f}")


if __name__ == "__main__":
    run()
