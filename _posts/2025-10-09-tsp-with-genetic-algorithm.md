---
title: "Traveling Salesman Problem with Genetic Algorithm"
date: 2025-10-09
categories: [Projects, Optimization, Genetic Algorithms]
tags: [Projects, TSP, Genetic Algorithm, Python, OpenRouteService, Folium, Jupyter]
---

# Solving the Traveling Salesman Problem with a Genetic Algorithm

The Traveling Salesman Problem (TSP) is a classic optimization challenge: given a list of cities, find the shortest possible route that visits each city exactly once and returns to the starting point. In my latest project, I tackled this problem for 9 Moroccan cities using a Genetic Algorithm (GA). Here's a deep dive into the project, how it works, and how you can try it yourself!

You can find the full code and resources in my GitHub repository: [github.com/ammarlouah/TSP_GA](https://github.com/ammarlouah/TSP_GA).

---

## What’s the Project About?

This project uses a Genetic Algorithm to solve a TSP instance for the following Moroccan cities:
- Khouribga (starting point)
- Settat
- Beni Mellal
- Tangier
- Errachidia
- Fes
- Rabat
- Casablanca
- Mohammedia

The solution:
- Fetches real-world city coordinates from OpenStreetMap.
- Uses the OpenRouteService API to compute realistic road distances between cities.
- Applies a Genetic Algorithm to find an optimized route.
- Visualizes the best route on an interactive map using Folium.

The result? A neat HTML map showing the best route and a JSON file with cached distance data for efficiency.

---

## What’s Inside the Repository?

The repository ([github.com/ammarlouah/TSP_GA](https://github.com/ammarlouah/TSP_GA)) contains:
- `generate_data.ipynb`: A Jupyter notebook to fetch city coordinates and compute distances, saving them to `city_distances.json`.
- `TSP_GA.ipynb`: The main notebook implementing the GA (population generation, fitness evaluation, selection, crossover, mutation, elitism) and generating the interactive map (`best_route_map.html`).
- `city_distances.json`: Cached distance matrix to avoid repeated API calls.
- `best_route_map.html`: Interactive map showing the best route found.
- `requirements.txt`: Python dependencies.
- `.env`: Local file (not tracked) for storing your OpenRouteService API key.
- `README.md`: Detailed setup and usage instructions.

---

## How Does the Genetic Algorithm Work?

The GA mimics natural evolution to find a good solution:
- **Population**: A set of random routes (potential solutions).
- **Fitness**: Evaluates each route based on total distance (shorter is better).
- **Selection**: Picks the best routes to "breed" the next generation.
- **Crossover**: Combines parts of two parent routes to create new ones.
- **Mutation**: Randomly tweaks routes to maintain diversity.
- **Elitism**: Preserves the best routes to ensure quality improves over generations.

You can tweak parameters like population size, number of generations, mutation rate, and more in the `TSP_GA.ipynb` notebook to experiment with performance.

---

## How to Run It on Your Machine

Want to try it out? Here’s how to get started:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ammarlouah/TSP_GA.git
   cd TSP_GA
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include `numpy`, `pandas`, `folium`, `requests`, `python-dotenv`, and `osmnx`.

4. **Get an OpenRouteService API Key**:
   - Sign up at [openrouteservice.org](https://openrouteservice.org) for a free API key.
   - Create a `.env` file in the project root with:
     ```
     API_KEY=your_openrouteservice_api_key_here
     ```

5. **Generate Distance Data**:
   - Open `generate_data.ipynb` in Jupyter Notebook or JupyterLab and run all cells.
   - This creates or updates `city_distances.json` with the distance matrix.

6. **Run the Genetic Algorithm**:
   - Open `TSP_GA.ipynb` and run all cells.
   - This reads `city_distances.json`, runs the GA, and generates `best_route_map.html`.

7. **View the Result**:
   - Open `best_route_map.html` in a web browser to see the interactive route map.
   - Or from the terminal:
     ```bash
     open best_route_map.html  # On Windows: start best_route_map.html
     ```

---

## Sample Output

After running the GA, you’ll get:
- **Interactive Map**: `best_route_map.html` shows the best route with city markers and a polyline.
- **Distance Matrix**: `city_distances.json` stores the distances for reuse.
- **Total Distance**: Printed in the notebook (varies due to GA randomness).

![Route Map](/assets/img/posts/tsp-with-genetic-algorithm/map.png)

---

## Tips and Troubleshooting

- **API Limits**: The free OpenRouteService plan has rate limits. If you hit them, add delays in `generate_data.ipynb` or reuse the cached `city_distances.json`.
- **Reproducibility**: The GA involves randomness. Set a fixed random seed in `TSP_GA.ipynb` for consistent results.
- **Adding Cities**: To include more cities, update `generate_data.ipynb` and ensure the distance matrix aligns with the GA code.
- **Parameter Tuning**: Experiment with GA parameters (e.g., `population_size`, `mutation_rate`) in `TSP_GA.ipynb` to optimize results.

---

## Get Involved

Want to contribute or have ideas? 
- Open an issue on the [GitHub repo](https://github.com/ammarlouah/TSP_GA) for bugs or suggestions.
- Experiment with different GA strategies (e.g., selection methods, crossover types) and share your findings!

---

## License

This project is licensed under the MIT License. Check the `LICENSE` file in the repository for details.

---

## Let’s Connect

Got questions or want to chat about the project? Reach out to me at [ammarlouah9@gmail.com](mailto:ammarlouah9@gmail.com) or open an issue on GitHub. Happy coding!
