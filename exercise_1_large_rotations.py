# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: queens
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Large rotations
# $
# % Define TeX macros for this document
# \def\vv#1{\boldsymbol{#1}}
# \def\mm#1{\boldsymbol{#1}}
# \def\R#1{\mathbb{R}^{#1}}
# \def\SO{SO(3)}
# \def\triad{\mm{\Lambda}}
# $
#
# We will use the large rotation framework implemented in the beam finite element input generator [**BeamMe**](https://github.com/beamme-py/beamme).
# BeamMe provides a `Rotation` class that encapsulates various representations of rotations (rotation matrices, rotation vectors, quaternions, etc.) and methods for converting between them, composing rotations, and applying rotations to vectors.
#
# Before solving the following exercises, have a look at the large rotation example notebook in the BeamMe repository: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/beamme-py/beamme/main?labpath=examples%2Fexample_1_finite_rotations.ipynb)

# %% [markdown]
# ## Exercises
#
# We need to import the relevant python packages and objects for the exercises:

# %%
import numpy as np
from beamme.core.rotation import Rotation

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.1:</strong>
#
#   Create a rotation object representing a rotation of 90 degrees about the $z$-axis and print the quaternion, rotation matrix, and rotation vector representations of this rotation.
# </div>

# %%
# Insert code for Exercise 1.1 here
rotation = Rotation([0.0, 0.0, 1.0], np.pi / 2)
print("Quaternion:", rotation.get_quaternion())
print("Rotation matrix:", rotation.get_rotation_matrix())
print("Rotation vector:", rotation.get_rotation_vector())

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.2:</strong>
#
#   Use the rotation from Exercise 1.1 to rotate the vector $\vv{a} = [1, 0, 0]^T$. Print the rotated vector.
# </div>

# %%
a = [1.0, 0.0, 0.0]

# Insert code for Exercise 1.2 here
rotated_a = rotation * a
print("Rotated vector:", rotated_a)

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.3:</strong>
#
#   Create two rotation objects: $\triad_1$ representing a rotation of $45^\circ$ degrees about the $x$-axis and $\triad_2$ representing a rotation of $30^\circ$ about the $y$-axis. Compose these two rotations, by first applying $\triad_1$ and then $\triad_2$. Extract and print the rotation vector corresponding to the composed rotation.
#
#   Show that a reordering of the how the rotations are applied leads to a different result, i.e., first applying $\triad_2$ and then $\triad_1$.
# </div>

# %%
# Insert code for Exercise 1.3 here
lambda_1 = Rotation([1.0, 0.0, 0.0], np.pi / 4)
lambda_2 = Rotation([0.0, 1.0, 0.0], np.pi / 6)
lambda_composed = lambda_2 * lambda_1
print("Composed rotation vector:\n", lambda_composed.get_rotation_vector())

lambda_composed_reordered = lambda_1 * lambda_2
print(
    "Reordered composed rotation vector:\n",
    lambda_composed_reordered.get_rotation_vector(),
)

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.4:</strong>
#
#   In this exercise, we investigate the inverse of a finite rotation and its properties.
#
#   Create a rotation object $\triad$ representing a rotation of $60^\circ$ about the axis
#   $[1, 1, 0]^T$.
#
#   1. Compute the inverse rotation $\triad^{-1}$ and print the rotation vectors of $\triad$ and $\triad^{-1}$. Comment on the relation between them.
#   2. Apply $\triad$ to a vector $\vv{a} = [1, 0.2, -0.1]^T$ and then apply the inverse
#       rotation $\triad^{-1}$ to the result. Verify that the original vector is recovered.
#   3. Verify that the composition $\triad^{-1}\triad$ (and $\triad\triad^{-1}$) corresponds
#       to the identity rotation by checking its rotation matrix representation.
#
#   *Hint:* You may use the `inv()` method of the `Rotation` class to compute the inverse rotation.
# </div>

# %%
angle = np.pi / 3
axis = [1.0, 1.0, 0.0]
a = [1.0, 0.2, -0.1]

# Insert code for Exercise 1.4 here
rotation = Rotation(axis, angle)
inverse_rotation = rotation.inv()
print("Original rotation vector:", rotation.get_rotation_vector())
print("Inverse rotation vector:", inverse_rotation.get_rotation_vector())
# The resulting rotation vectors are pointing in opposite directions with the same magnitude.

rotated_a = rotation * a
restored_a = inverse_rotation * rotated_a
print("Restored vector a:", restored_a)

composition_1 = inverse_rotation * rotation
composition_2 = rotation * inverse_rotation
print("Composition 1 rotation matrix:", composition_1.get_rotation_matrix())
print("Composition 2 rotation matrix:", composition_2.get_rotation_matrix())

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.5:</strong>
#
#   In the lecture notes, we discussed the difference between additive and multiplicative rotation vector increments. Combining (2.14) and (2.15) gives the following relations:
#   $$
#   \triad(\vv{\psi}_2) =
#   \triad(\vv{\psi}_1 + \Delta \vv{\psi}) =
#   \triad(\Delta \vv{\theta})  \triad(\vv{\psi}_1) =
#   \triad(\vv{\psi}_1) \triad(\Delta \vv{\Theta}).
#   $$
#   Here, $\Delta \vv{\psi}$ is the additive rotation vector increment, while $\Delta \vv{\theta}$ and $\Delta \vv{\Theta}$ are the multiplicative rotation vector increments applied from the left and right, respectively.
#
#   The initial rotation vector $\vv{\psi}_1 = [0.1, 0.2, 0.3]^T$ and additive increment $\Delta \vv{\psi} = [0.01, -0.02, 0.03]^T$ are given.
#
#   1. Compute $\vv{\psi}_2$
#   2. Compute the multiplicative increments $\Delta \vv{\theta}$ and $\Delta \vv{\Theta}$.
#   3. Verify the relation (2.16), i.e., $\Delta \vv{\theta} = \triad(\vv{\psi}_1) \Delta \vv{\Theta}$.
#   4. Check if the relation (2.22), $\delta \vv{\psi} = \mm{T}(\vv{\psi}) \delta \vv{\theta}$ holds for the given values. Comment on the result.
#
#   *Hint**: You may use the `get_transformation_matrix()` method of the `Rotation` class to compute the transformation matrix $\mm{T}(\vv{\psi})$.
# </div>

# %%
psi_1 = np.array([0.1, 0.2, 0.3])
delta_psi = np.array([0.01, -0.02, 0.03])

# Insert code for Exercise 1.5 here
psi_2 = psi_1 + delta_psi
print("psi_2:", psi_2)

rotation_1 = Rotation.from_rotation_vector(psi_1)
rotation_2 = Rotation.from_rotation_vector(psi_2)

left_relative_rotation = rotation_2 * rotation_1.inv()
delta_theta_left = left_relative_rotation.get_rotation_vector()
print("Left multiplicative rotation vector increment:", delta_theta_left)

right_relative_rotation = rotation_1.inv() * rotation_2
delta_theta_right = right_relative_rotation.get_rotation_vector()
print("Right multiplicative rotation vector increment:", delta_theta_right)

print(
    "Left multiplicative rotation vector increment via (2.16):",
    rotation_1 * delta_theta_right,
)

delta_psi_via_transformation = rotation_1.get_transformation_matrix() @ delta_theta_left
print("Delta psi via transformation matrix:", delta_psi_via_transformation)
# The result is close to delta_psi, but not exactly. The relation only holds for infinitesimal increments, however, the given increment is finite.
