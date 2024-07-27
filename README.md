# The Planets
## Planetary stability thought experiment

We posed the following thought experiment to Claude.ai
>You are familiar with kepler's laws of motion and the barnes-hut and similar algorithms for multi-body calculations. We wish to study the stability of a multi-planet system where the planets occupy the same circular orbit at symmetrical positions around the sun. For initial conditions consider typical solar mass for the sun and typical earth mass for the planets. Consider numbers of planets from 2 to 200. Perform numerical calculations for a period of 1000 years. Apply a small angular perturbation to one planet. Does this perturbation grow, remain stable or decrease? Plot the results as a function of planet number.

After a few clarifying prompts, Claude very generoulsy wrote the script we called mutlit_planet_stability.py You can view our (Roger Carr's and my) dialog with Claude [here](https://claude.ai/chat/5e2c6767-b16d-4833-86a6-f2bfede7fa88).

Jack Pham was able to make the generated code run on Google Colab with only a very few edits.
