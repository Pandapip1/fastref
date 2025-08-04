#!/usr/bin/env python
import glob
import mdtraj as md
import numpy as np
from enspara.cluster import KCenters, KHybrid
from enspara.msm import MSM
from fast import AdaptiveSampling
from fast.md_gen.gromacs import Gromacs, GromacsProcessing
from fast.msm_gen import ClusterWrap
from fast.analysis.rmsd import RMSDWrap
from fast.sampling import rankings, scalings
from fast.submissions.slurm_subs import SlurmSub, SlurmWrap
from fast import SaveWrap
from fast.analysis.pockets import ResiduePockets


def entry_point():

    ###########################################
    #            define parameters            #
    ###########################################

    # SIMULATION PARAMETERS

    input_dir = "./inputs"  # CHANGE IF NEEDED
    # input_dir should be a path to a directory that contains
    # a topol.top file from equilibration
    # an mdp file that contains the simulation parameters
    # a pdb file of the protein with hydrogens (i.e. equilibrated starting structure - no solvent)
    # a start.gro file of the equilibrated system

    q_name = "a5000"  # qr6 or a5000 # CHANGE IF NEEDED
    constraint = "nvidiagpu"
    # q_name -- The name of the queue you are going to submit to
    # currently we have support for LSF and SLURM queues, as well as submitting on a laptop/desktop.

    # gpu_info = '"num=1:gmodel=QuadroRTX6000"'
    # Name of the GPU, which is needed on our queuing system to select a GPU specific node
    # This is important for the code to be able to generate the appropriate header for
    # submitting to your queing system
    # For example, this will end up writing out #BSUB -gpu "num=1:gmodel=TeslaP100_PCIE_16GB"

    n_cpus_gromacs = 12
    n_gpus_gromacs = 1

    sim_name = "TODO_RUNFASTALT_SIM_NAME"  # CHANGE!
    # Name to give the job on your queueing system

    top_filename = f"{input_dir}/{sim_name}.top"
    mdp_filename = (
        "/mnt/pure/bowmanlab/WG-shahlo/projects/Ab_dynamics/simulations/pr-50ns.mdp"
    )
    print(top_filename, mdp_filename)
    ndx_filename = None  # f"{input_dir}/index.ndx" # change if groups are other than protein / system
    # print(top_filename, mdp_filename, ndx_filename)

    # Path to gromacs installation
    gromacs_gmxrc_path = "./GMXRC"

    itp_files = None

    pbc = "mol"
    ur = "compact"
    align_group = "Protein"
    output_group = "Prot-Masses"
    center_group = "Protein"  # only use when pbc is set to cluster.
    # Options that will be used by gromacs' trjconv to align trajectories between rounds of FAST

    # CLUSTERING PARAMETERS
    cluster_radius = 0.15
    # Radius to be used for KCenters clustering
    prot_masses = f"{input_dir}/{sim_name}-prot-masses.pdb"

    atom_indices = f"{input_dir}/atom_indices.dat"
    # The atom indices (with respect to prot_masses) you want to use to cluster between rounds of FAST
    # Backbone atoms are a solid default choice.

    n_cpus_clustering = 24

    # save states
    save_routine = "full"
    # The type of states to save. Three options: 1) 'masses' saves
    #    only in the centers_masses, 2) 'restarts' saves only the
    #    restarts, and 3) 'full' saves both.

    save_centers = "all"
    # The indicator for the set of centers to save. Four options:
    #    1) 'all' will save every center, 2) 'none' will not save any centers,
    #    3) 'restarts' will only save the centers to use for
    #    restarting simulations, and 4) 'auto' will only save new states
    #    that were discovered in previous round of sampling.

    save_xtc_centers = True
    # Option to save all the cluster centers as an xtc file.

    n_cpus_save = 24

    # RANKING PARAMETERS
    directed_scaling = scalings.feature_scale(maximize=True)
    # here we're trying to maximize the contacts

    distance_metric = md.rmsd
    # This will ultimately be used to discourage FAST from choosing geometrically similar
    # states when choosing states for further simulation.
    width = 0.36
    # This is the gaussian spread that is used to distinguish between states.

    # ADAPTIVE SAMPLING PARAMETERS

    starting_structure = f"{input_dir}/{sim_name}-start.gro"
    submission_obj = SlurmSub("amdcpu", n_cpus=24)
    # This creates a script to submit clustering and analysis jobs to the appropriate
    # nodes in your queue with the appropriate keyword arguments.
    # This may require customization for your specific queuing system. Please
    # explore submissions/lsf_subs.py, submissions/os_sub.py, submissions/slurm_subs.py
    # for examples on how to build a class that is specific to your queuing system.

    n_gens = 5
    # Number of rounds or "generations" of FAST to run
    n_kids = 10
    # Number of simulations to run in each generation of FAST

    update_freq = 1
    # The number of generations between a full reclustering of states and
    #    analysis of cluster centers. Defaults to never reclustering
    #    (continually adds new cluster centers without changing previously
    #    discovered centers).

    continue_prev = False
    # If you are continuing a FAST run, you can set this to True

    output_dir = "FAST_RMSD"
    # Name of directory to write to store FAST output

    ############################################
    #            initialize objects            #
    ############################################

    # SIMULATIONS OBJECTS
    gro_submission = SlurmSub(
        q_name, n_tasks=n_cpus_gromacs, job_name=sim_name, gpus=1, constraint=constraint
    )

    gro_processing = GromacsProcessing(
        align_group=align_group,
        output_group=output_group,
        pbc=pbc,
        ur=ur,
        index_file=ndx_filename,
    )
    # In this specific example, 10 is the index for protein. Check your
    # .ndx file and select the index that chooses protein.

    sim_obj = Gromacs(
        top_file=top_filename,
        mdp_file=mdp_filename,
        n_cpus=n_cpus_gromacs,
        n_gpus=n_gpus_gromacs,
        processing_obj=gro_processing,
        submission_obj=gro_submission,
        pin="on",
        setup_path=gromacs_gmxrc_path,
        itp_files=itp_files,
    )

    # CLUSTERING OBJECT
    base_clust_obj = KHybrid(metric=md.rmsd, cluster_radius=cluster_radius)
    clust_obj = ClusterWrap(
        base_struct=prot_masses,
        base_clust_obj=base_clust_obj,
        atom_indices=atom_indices,
        n_procs=n_cpus_clustering,
    )

    # SAVE STATE OBJECT
    save_state_obj = SaveWrap(
        save_routine=save_routine,
        centers=save_centers,
        n_procs=n_cpus_save,
        save_xtc_centers=save_xtc_centers,
    )

    # ANALYSIS OBJECT
    anal_obj = RMSDWrap(prot_masses)

    # RANKING OBJECT
    ranking_obj = rankings.FAST(
        directed_scaling=directed_scaling, distance_metric=distance_metric, width=width
    )

    # SAMPLING OBJECT
    a = AdaptiveSampling(
        starting_structure,
        n_gens=n_gens,
        n_kids=n_kids,
        sim_obj=sim_obj,
        cluster_obj=clust_obj,
        save_state_obj=save_state_obj,
        analysis_obj=anal_obj,
        ranking_obj=ranking_obj,
        continue_prev=continue_prev,
        update_freq=update_freq,
        sub_obj=submission_obj,
        output_dir=output_dir,
        q_check_obj=SlurmWrap(),
        q_check_obj_sim=SlurmWrap(),
    )

    ##############################################
    #                run sampling                #
    ##############################################

    # run
    a.run()


if __name__ == "__main__":
    entry_point()
