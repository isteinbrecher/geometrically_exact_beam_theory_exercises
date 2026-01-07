"""This file contains functionality for the lecture lab."""

import subprocess
from pathlib import Path

import numpy as np
import yaml
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.four_c.header_functions import set_header_static, set_runtime_output
from beamme.four_c.input_file import InputFile
from beamme.four_c.run_four_c import clean_simulation_directory
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def create_beam_mesh_line_2d(
    mesh, beam_class, material, start_point, end_point, **kwargs
):
    """Create a 2D line beam mesh."""
    start_point_3d = np.array([start_point[0], start_point[1], 0.0])
    end_point_3d = np.array([end_point[0], end_point[1], 0.0])
    beam_set = create_beam_mesh_line(
        mesh,
        beam_class,
        material,
        start_point_3d,
        end_point_3d,
        set_nodal_arc_length=True,
        **kwargs,
    )
    mesh.add(
        BoundaryCondition(
            beam_set["line"],
            {
                "NUMDOF": 6,
                "ONOFF": [0, 0, 1, 1, 1, 0],
                "VAL": [0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    return beam_set


def create_boundary_condition_2d(
    mesh,
    geometry_set,
    *,
    directions=None,
    bc_type=None,
    values=None,
    linear_increase=True,
):
    """Create a boundary condition for plane beams."""

    if values is not None and not len(directions) == len(values):
        raise ValueError("directions and values must have the same length")

    if values is None:
        values = [0.0] * len(directions)

    value_bc = [0, 0, 0, 0, 0, 0]
    on_off = [0, 0, 0, 0, 0, 0]
    if "x" in directions:
        on_off[0] = 1
        value_bc[0] = values[directions.index("x")]
    if "y" in directions:
        on_off[1] = 1
        value_bc[1] = values[directions.index("y")]
    if "theta" in directions:
        on_off[5] = 1
        value_bc[5] = values[directions.index("theta")]

    if bc_type == "dirichlet":
        bc_type_value = bme.bc.dirichlet
        function_string = (
            "SPACE_TIME" if geometry_set.geometry_type == bme.geo.point else "TIME"
        )
    elif bc_type == "neumann":
        bc_type_value = bme.bc.neumann
        function_string = "TIME"
    else:
        raise ValueError("bc_type must be either 'dirichlet' or 'neumann'")

    if not linear_increase:
        function_bc = [0, 0, 0, 0, 0, 0]
    else:
        function_load = Function([{f"SYMBOLIC_FUNCTION_OF_{function_string}": "t"}])
        mesh.add(function_load)
        function_bc = [function_load] * 6

    mesh.add(
        BoundaryCondition(
            geometry_set,
            {"NUMDOF": 6, "ONOFF": on_off, "VAL": value_bc, "FUNCT": function_bc},
            bc_type=bc_type_value,
        )
    )


def run_four_c(
    *,
    mesh=None,
    simulation_name=None,
    total_time=1.0,
    n_steps=1,
    tol=1e-10,
):
    """Run a 4C simulation with given parameters."""

    # Setup the input file with mesh and parameters.
    input_file = InputFile()
    input_file.add(mesh)
    set_header_static(
        input_file,
        total_time=total_time,
        n_steps=n_steps,
        max_iter=20,
        tol_residuum=1.0,
        tol_increment=tol,
        create_nox_file=False,
        predictor="TangDis",
    )
    set_runtime_output(
        input_file,
        output_solid=False,
        output_stress_strain=False,
        btsvmt_output=False,
        btss_output=False,
        output_triad=True,
        every_iteration=False,
        absolute_beam_positions=True,
        element_owner=True,
        element_gid=True,
        element_mat_id=True,
        output_energy=False,
        output_strains=True,
    )
    input_file["IO/RUNTIME VTK OUTPUT/BEAMS"]["MATERIAL_FORCES_GAUSSPOINT"] = True
    input_file["IO/MONITOR STRUCTURE DBC"] = {
        "INTERVAL_STEPS": 1,
        "WRITE_CONDITION_INFORMATION": True,
        "FILE_TYPE": "yaml",
    }

    # Dump the file to disc.
    simulation_directory = Path.cwd() / simulation_name
    input_file_path = simulation_directory / f"{simulation_name}.4C.yaml"
    clean_simulation_directory(simulation_directory)
    input_file.dump(input_file_path)

    # Create file that maps the global node IDs to the beam arc length.
    arc_length_file_path = simulation_directory / f"{simulation_name}_arc_length.yaml"
    data = {}
    for element in mesh.elements:
        data[element.i_global] = [
            float(element.nodes[node_index].arc_length) for node_index in (0, -1)
        ]
    with open(arc_length_file_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Run the simulation and process the results line by line.
    with open(simulation_directory / f"{simulation_name}.log", "w") as logfile:
        # Command to run 4C
        four_c_exe = "/data/a11bivst/dev/4C/release/4C"
        command = [four_c_exe, input_file_path.absolute(), simulation_name]

        # Start simulation
        print("Start simulation")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout (optional)
            text=True,  # get str instead of bytes
            bufsize=1,  # line-buffered
            cwd=simulation_directory,
        )

        # Process the output line by line
        nonlinear_solver_step_count = 0
        is_error = False
        finished = False
        for line in process.stdout:
            line = line.rstrip("\n")

            # Write line to logfile
            logfile.write(line + "\n")

            # Flush file so log is always up to date
            logfile.flush()

            # Process the line however you want
            if "Nonlinear Solver Step" in line:
                nonlinear_solver_step_count = int(line.split(" ")[4])
            elif "||F||" in line:
                if not nonlinear_solver_step_count == 0:
                    residuum = float(line.split(" ")[10])
                    print(
                        f"  Nonlinear Solver Step {nonlinear_solver_step_count}: Residuum = {residuum:.3e}"
                    )
            elif "Finalised step" in line:
                split = line.split(" ")
                step = int(split[2])
                time = float(split[7])
                print(f"Finished time step {step} for time {time:.3e}")
            elif "OK (0)" in line:
                finished = True
            elif (
                "========================================================================="
                in line
                and not finished
            ):
                if is_error:
                    print(line)
                is_error = not is_error

            if is_error:
                print(line)

        _return_code = process.wait()
