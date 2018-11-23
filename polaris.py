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


def prepare_input(calc_params, strengths):
    tpl = """
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

    &scf
    """

    """
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
      1.0e-10,1.0e-4,1.0e-4
    """
    template = Template(textwrap.dedent(tpl))
    directions = "X Y Z".split()
    product = list(it.product(strengths, directions))
    job_input = "\n".join(
        [template.render(direction=d, strength=s, calc=calc_params)
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


def finit_diff(i, j, d_one, d_two, d_four, strength):
    alpha = (256*d_one[j][i] - 40*d_two[j][i] + d_four[j][i])/(360*strength)
    return alpha


def central_difference(i, j, diff_, strength):
    return diff_ / (2*strength)


def two_fields(i, j, d_one, d_two, strength):
    # alpha = 
    pass


def run():
    # with open("05_ffpt_test.out") as handle:
        # text = handle.read()
    # ens, dpms = parse_log(text)

    # job = prepare_input(0.001)
    # print(job)

    # Nur Startstärke und Anzahl der Felder auswählen,
    # die Funktion macht dann den Rest.

    strengths = 0.002 * np.array((1, -1, 2, -2, 4, -4))
    fields = strengths.size
    calc_params = {
        # "xyz": "/home/carpx/Arbeit/polaris/ammoniak/backup/symmetry.xyz",
        # "xyz": "/scratch/molcas_jobs/nh3_inversion/backup/01_relaxed_scan/nh3_inversion.Opt.15.xyz",
        "xyz": "/scratch/polarisierbarkeit/geometrien/formaldehyd.xyz",
        # "basis": "ano-rcc-vdzp",
        "basis": "aug-cc-pvdz",
        "charge": 0,
        "spin": 1,
        "fileorb": "/scratch/molcas_jobs/nh3_inversion/backup/05_casscf_pes/nh3_inversion.15.RasOrb",
        "ciroot": 5,
    }
    job_input, job_order = prepare_input(calc_params, strengths)
    job_order_str = " ".join([f"({s:.3f} {d})" for s, d in job_order])
    # text = run_molcas(job_input)
    # with open("job.last", "w") as handle:
        # handle.write(text)
    # return
    fn = "job.last.scf.formaldehyd"
    with open(fn) as handle:
        text = handle.read()
    ens, dpms = parse_log(text)
    np.savetxt("energies", ens)
    np.savetxt("dipoles", dpms)
    ens = ens.reshape(strengths.size, 3, -1)
    # Drop the total component
    dpms = dpms[:,:3]
    dpms = dpms.reshape(strengths.size, -1, 3)
    # Shape (No. of electric fields, No. of states * 3, dipole components)
    # dpms = dpms.reshape(strengths.size, 3, -1, 4)

    # Convert from Debye to a.u.
    dpms /= 2.5418


    sum1 = dpms[0] + dpms[1]
    diff1 = dpms[0] - dpms[1]
    diff2 = dpms[2] - dpms[3]
    alphs = ((2/3)*diff1 - (1/12)*diff2)/0.002
    # (XYZ), (ciroot), (DPM components)
    diff1 = diff1.reshape(3, -1, 3)

    cd = diff1/0.002
    print("cd")
    print(cd)
    ax = "XYZ"
    import pdb; pdb.set_trace()
    for i in range(3):
        print(f"alpha_{ax[i]}{ax[i]}")
        a = cd[i][:,i]
        print(a)
    return
    avg_alphas = np.sum(cd, axis=1)*1/3
    import pdb; pdb.set_trace()

    # axes = (0, 1, 2)
    # tensor_axes = it.combinations_with_replacement(axes, 2)
    # for i, j in tensor_axes:
        # fd = finit_diff(i, j, d_one, d_two, d_four, 0.001)
        # print(i, j, ":", fd)


if __name__ == "__main__":
    run()
