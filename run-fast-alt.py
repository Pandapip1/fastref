#!/usr/bin/env python

#### Resources (pre-import)

n_cpus_preimport = 16

# Imports
import os
os.environ["NUMEXPR_MAX_THREADS"] = n_cpus_preimport
import mdtraj as md
from enspara.cluster import KHybrid
from fast import AdaptiveSampling, SaveWrap
from fast.md_gen.gromacs import Gromacs, GromacsProcessing
from fast.msm_gen import ClusterWrap
from fast.sampling import rankings, scalings
from fast.submissions.slurm_subs import SlurmSub, SlurmWrap
from fast.analysis.pockets import PocketWrap
from fast.analysis.pocketminer import PMExpectedVolumeWrap
from pypocketminer.models.mqa_model import MQAModel

#### General

# See inputs/README.md
input_dir = "./inputs"

# The output directory. Generally, keep this as-is.
output_dir = "./FAST_OUT"

# This file is sourced before every run. Set up the environment using this.
env_setup_path = "./env"

#### Resources

# The SLURM queues to use for CPU / GPU work
normal_queue = "amdcpu"
gpu_queue = "a5a,a5000,p100,qr6"

# The number of threads to request for parallelizable tasks
n_cpus = n_cpus_preimport
n_cpus_cpuheavy = 64

# The number of GPUs to request for GPU-able tasks
n_gpus = 1

#### Sampling

# Number of rounds or "generations" of FAST to run
n_gens = 5

# Number of simulations to run in each generation of FAST
n_kids = 10

#### Trjconv Alignment
processing_obj = GromacsProcessing(
    # TODO: Documentation
    align_group="Protein",
    # TODO: Documentation
    output_group="Prot-Masses",
    # TODO: Documentation
    pbc="mol",
    # TODO: Documentation
    ur="compact",
    # Change if groups are other than protein / system
    index_file=None,  # Optional: e.g., f"{input_dir}/index.ndx"
)

#### Clustering
base_clust_obj = KHybrid(
    # TODO: Documentation
    metric=md.rmsd,
    # TODO: Documentation
    cluster_radius=0.15,
)

# The number of generations between a full reclustering of states and
#    analysis of cluster centers. Defaults to never reclustering
#    (continually adds new cluster centers without changing previously
#    discovered centers).
update_freq = 1

#### State Saving
save_state_obj = SaveWrap(
    # The type of states to save. Three options: 1) 'masses' saves
    #    only in the centers_masses, 2) 'restarts' saves only the
    #    restarts, and 3) 'full' saves both.
    save_routine="full",
    # The indicator for the set of centers to save. Four options:
    #    1) 'all' will save every center, 2) 'none' will not save any centers,
    #    3) 'restarts' will only save the centers to use for
    #    restarting simulations, and 4) 'auto' will only save new states
    #    that were discovered in previous round of sampling.
    centers="all",
    # Option to save all the cluster centers as an xtc file.
    save_xtc_centers=True,
    n_procs=n_cpus,
)

#### Analyses

# The sims to run
sim_names = ["1EXM", "1JWP", "1NEP"]

# The geometric components to use
analyses = {
    "ligsite": {
        "ranking": rankings.FAST(
            # Maximize pocket volume
            directed_scaling=scalings.feature_scale(maximize=True),
            # Use RMSD to compare distances
            distance_metric=md.rmsd,
            # This is the gaussian spread that is used to distinguish between states.
            width=0.36,
        ),
        "analysis": PocketWrap(
            # Minimum rank for defining a pocket element. Ranges from 1-7, 1
            # being very shallow and 7 being a fully enclosed pocket element.
            min_rank=6,
            # The radius of the grid point to probe for pocket elements.
            probe_radius=0.14,
            # The minimum number of adjacent pocket elements to consider a
            # true pocket. Trims pockets smaller than this size.
            min_cluster_size=3,
            n_cpus=n_cpus_cpuheavy,
        ),
    },
    "pocketminer_volume": {
        "ranking": rankings.FAST(
            # Maximize pocket volume
            directed_scaling=scalings.feature_scale(maximize=True),
            # Use RMSD to compare distances
            distance_metric=md.rmsd,
            # This is the gaussian spread that is used to distinguish between states.
            width=0.36,
        ),
        "analysis": PMExpectedVolumeWrap(
            # Path to pretrained PocketMiner checkpoint
            nn_path=f"{input_dir}/pocketminer_pretrained/pocketminer.index",
            # The model corresponding to the checkpoint
            model=MQAModel(
                node_features=(8, 50),
                edge_features=(1, 32),
                hidden_dim=(16, 100),
                num_layers=4,
                dropout=0.1,
            ),
            # 1 for simple sum; 2 for sum of sqaures.
            # You can technically put whatever power you want here
            power=1,
        ),
    },
}


if __name__ == "__main__":
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    sim_name = sim_names[task_id % len(sim_names)]
    anl_names = list(analyses.keys()).sort()
    anl_name = anl_names[task_id // len(sim_names)]
    analysis_objs = analyses[anl_name]

    os.makedirs(f"{output_dir}/{name}", exist_ok=True)
    continue_prev = False
    for i in range(n_kids):
        continue_prev = os.path.exists(f"{output_dir}/{anl_name}/{sim_name}/gen0/kid{i}/frame0.xtc")
        if continue_prev: break
    AdaptiveSampling(
        # Generated from equilibration
        # See inputs/README.md
        f"{input_dir}/{sim_name}-start.gro",
        n_gens=n_gens,
        n_kids=n_kids,
        sim_obj=Gromacs(
            # Generated from equilibration
            # See inputs/README.md
            top_file=f"{input_dir}/{sim_name}.top",
            # Copied from "/mnt/pure/bowmanlab/WG-shahlo/projects/Ab_dynamics/simulations/pr-50ns.mdp"
            # TODO: Documentation
            mdp_file=f"{input_dir}/pr-50ns.mdp",
            # TODO: Documentation
            itp_files=None,
            # TODO: ???
            pin="on",
            n_cpus=n_cpus,
            n_gpus=n_gpus,
            setup_path=env_setup_path,
            processing_obj=processing_obj,
            submission_obj=SlurmSub(
                gpu_queue,
                n_tasks=n_cpus,
                gpus=n_gpus,
                nice="10000",
            ),
        ),
        cluster_obj=ClusterWrap(
            base_clust_obj=base_clust_obj,
            # Generated from equilibration
            # See inputs/README.md
            base_struct=f"{input_dir}/{sim_name}-prot-masses.pdb",
            # The atom indices (with respect to prot_masses) you want to use to cluster between rounds of FAST
            # Backbone atoms are a solid default choice.
            # Generated from equilibration
            # See inputs/README.md; inputs/save_inds.py
            atom_indices=f"{input_dir}/{sim_name}_atom_indices.dat",
            n_procs=n_cpus_cpuheavy,
        ),
        save_state_obj=save_state_obj,
        continue_prev=continue_prev,
        update_freq=update_freq,
        q_check_obj=SlurmWrap(),
        q_check_obj_sim=SlurmWrap(),
        sub_obj=SlurmSub(
            normal_queue,
            n_cpus=n_cpus_cpuheavy,
            job_name=f"SlurmSub_{sim_name}_{anl_name}_AdaptiveSampling",
            nice="10000",
        ),
        analysis_obj=analysis_objs["analysis"],
        ranking_obj=analysis_objs["ranking"],
        output_dir=f"{output_dir}/{anl_name}/{sim_name}",
    ).run()
