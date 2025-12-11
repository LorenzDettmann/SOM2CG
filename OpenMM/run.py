from openmm.unit import nanometer, kelvin, picosecond, femtosecond, bar
from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import Simulation, StateDataReporter, GromacsGroFile, PDBFile
import martini_openmm as martini
from mdtraj.reporters import XTCReporter

# configuration
INPUT_DIR = "INIT_CG"
GRO_FILE = f"{INPUT_DIR}/solvated.gro"
TOP_FILE = f"{INPUT_DIR}/topol.top"
EPSILON_R = 15.0
TEMP = 298.15 * kelvin
PRESSURE = 1.0 * bar
DT = 20 * femtosecond
DT_NVT = 2 * femtosecond
FRICTION_COEFF = 1.0 / picosecond

# steps
NSTEPS_NVT = 50000       # 0.1 ns
NSTEPS_NPT = 500000      # 10 ns
NSTEPS_RUN = 500000      # 10 ns

def create_simulation_environment(topology, system, coords, box, DT, platform_name="CPU"):
    """helper function to create a simulation environment"""
    integrator = LangevinMiddleIntegrator(TEMP, FRICTION_COEFF, DT)
    integrator.setRandomNumberSeed(0)
    
    properties = {}
    if platform_name == "CUDA":
        properties = {'Precision': 'mixed'}
        
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(topology, system, integrator, platform, properties)
    
    simulation.context.setPositions(coords)
    simulation.context.setPeriodicBoxVectors(*box)
    
    return simulation

def run_pipeline():
    platform_name = "CUDA" # or CPU

    print(f"Starting simulation on {platform_name}...")

    # load initial coordinates
    conf = GromacsGroFile(GRO_FILE)
    box_vectors = conf.getPeriodicBoxVectors()
    positions = conf.getPositions()

    ############################################################################
    ### 1. Energy Minimization (constraints defined as bonds)
    ############################################################################
    print("\n--- Phase 1: Energy Minimization ---")
    
    top_em = martini.MartiniTopFile(
        TOP_FILE,
        periodicBoxVectors=box_vectors,
        defines={'min': '1'},
        epsilon_r=EPSILON_R,
    )
    system_em = top_em.create_system(nonbonded_cutoff=1.1 * nanometer)
    
    sim_em = create_simulation_environment(top_em.topology, system_em, positions, box_vectors, DT, platform_name)

    print("Minimizing energy...")
    sim_em.minimizeEnergy(maxIterations=5000, tolerance=100)
    
    # extract positions and box for equilibration
    state = sim_em.context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions()
    
    print(f"Minimized energy: {state.getPotentialEnergy()}")
    PDBFile.writeFile(sim_em.topology, positions, open('em.pdb', 'w'))
    
    # clean up
    del sim_em, system_em, top_em

    ############################################################################
    ### 2. Setup for Equilibration and Production (Standard Topology)
    ############################################################################
    
    print("\n--- Setup for Production Topology (Constraints active) ---")
    
    top_run = martini.MartiniTopFile(
        TOP_FILE,
        periodicBoxVectors=box_vectors,
        defines=None, # normal topology
        epsilon_r=EPSILON_R,
    )
    
    # create system for NVT
    system_run = top_run.create_system(nonbonded_cutoff=1.1 * nanometer)

    sim_nvt = create_simulation_environment(top_run.topology, system_run, positions, box_vectors, DT_NVT, platform_name)
    
    # generate velocities
    sim_nvt.context.setVelocitiesToTemperature(TEMP)

    ############################################################################
    ### 3. NVT Equilibration
    ############################################################################
    print(f"\n--- Phase 2a: NVT Equilibration ({NSTEPS_NVT * DT_NVT.value_in_unit(picosecond) / 1000} ns) ---")
    
    sim_nvt.reporters.append(StateDataReporter("nvt.log", 1000, 
                                               step=True, potentialEnergy=True, 
                                               temperature=True, volume=True, speed=True))
    
    sim_nvt.step(NSTEPS_NVT)
    
    # save NVT state
    positions = sim_nvt.context.getState(getPositions=True).getPositions()
    velocities = sim_nvt.context.getState(getVelocities=True).getVelocities()
    PDBFile.writeFile(sim_nvt.topology, positions, open('nvt.pdb', 'w'))
    
    # clear reporters
    sim_nvt.reporters.clear()

    ############################################################################
    ### 4. NPT Equilibration
    ############################################################################
    print(f"\n--- Phase 2b: NPT Equilibration ({NSTEPS_NPT * DT.value_in_unit(picosecond) / 1000} ns) ---")
    
    # Add Barostat to the existing system
    barostat = MonteCarloBarostat(PRESSURE, TEMP, 10)
    system_run.addForce(barostat)
    
    # reinitialize the context to include the new Force
    sim_run = create_simulation_environment(top_run.topology, system_run, positions, box_vectors, DT, platform_name)
    sim_run.context.setVelocities(velocities)
    sim_run.context.reinitialize(preserveState=True)

    # reporter for NPT
    sim_run.reporters.append(StateDataReporter("npt.log", 5000, 
                                               step=True, potentialEnergy=True, 
                                               temperature=True, volume=True, density=True, 
                                               speed=True))
    
    sim_run.step(NSTEPS_NPT)
    
    sim_run.saveCheckpoint('npt.chk')
    positions = sim_run.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(sim_run.topology, positions, open('npt.pdb', 'w'))
    
    # clear reporter
    sim_run.reporters.clear()

    ############################################################################
    ### 5. Production Run
    ############################################################################
    print(f"\n--- Phase 3: Production Run ({NSTEPS_RUN * DT.value_in_unit(picosecond) / 1000} ns) ---")
    sim_run.context.setTime(0.0)
    sim_run.currentStep = 0
    
    # xtc reporter
    xtc_reporter = XTCReporter('run.xtc', 50000) # every 1 ns (assuming 20fs step)
    sim_run.reporters.append(xtc_reporter)
    
    # log file reporter
    sim_run.reporters.append(StateDataReporter("run.log", 5000,
                                               step=True, potentialEnergy=True, 
                                               totalEnergy=True, temperature=True, 
                                               volume=True, density=True, speed=True))

    sim_run.step(NSTEPS_RUN)

    positions = sim_run.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(sim_run.topology, positions, open('run.pdb', 'w'))
    
    print("Simulation finished.")

if __name__ == "__main__":
    run_pipeline()
