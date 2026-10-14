# Match-Day Nutrition with QAOA

This project uses the Quantum Approximate Optimization Algorithm (QAOA) to solve a match-day nutrition problem.

The goal is to select exactly 14 servings from eight dining-hall foods while approaching targets for calories, protein, carbohydrates, fat, and potassium, with an additional penalty for sodium.

## Foods

The model includes:

- Eggs
- Oatmeal
- Chicken
- Rice
- Vegetables
- Bananas
- Peanut butter
- Pizza

Each food can have between zero and three servings. This gives 65,536 possible menus.

## Method

The project follows these steps:

1. Define a nutrition cost function.
2. Find the exact solution using classical brute force.
3. Convert the problem into a QUBO.
4. Convert the QUBO into a 16-qubit Ising Hamiltonian.
5. Run QAOA with circuit depths \(p=1\), \(p=2\), and \(p=3\).
6. Compare the QAOA results with the exact classical result.

## Results

The Hamiltonian ground state matches the brute-force optimum, confirming that the quantum encoding is correct.

As the QAOA depth increases, the probability of finding good and feasible menus improves. At \(p=3\), the best sampled solution ranks sixth among all 65,536 possible menus.

Classical brute force is still much faster for this small problem, so this project does not claim a quantum speedup.

The results also show that the optimal menu depends strongly on the selected nutrient weights and constraint penalties.

## Run the Project

Install the required packages:

```bash
pip install numpy scipy matplotlib qiskit qiskit-optimization jupyterlab
```

Open the notebook:

```bash
jupyter lab nutrition_qaoa.ipynb
```

Then run all cells from top to bottom.

## Note

The nutritional targets are created for demonstration only. This project is not dietary or medical advice.
