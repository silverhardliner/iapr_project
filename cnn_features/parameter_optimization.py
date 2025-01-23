import json
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.optimize import differential_evolution

from clustering import get_pieces_per_image, create_true_labels, prepare_features, perform_clustering, evaluate_solution_clustering
from cnn_features import extract_features, get_solution_features
from solution_to_pieces import get_solution_pieces

def objective_function(params, features_per_image, true_labels):
    """Optimization objective function"""
    eps, min_samples, n_components = params
    min_samples = int(min_samples)  # Convert to integer
    n_components = int(n_components)  # Convert to integer
    
    print(f"\nTrying parameters:")
    print(f"  eps: {eps:.3f}")
    print(f"  min_samples: {min_samples}")
    print(f"  n_components: {n_components}")
    
    # Prepare features with PCA and scaling
    all_features_scaled, pca, scaler = prepare_features(features_per_image, n_components)

    # Perform clustering with current parameters
    predicted_labels = perform_clustering(all_features_scaled, eps, min_samples)

    # Calculate scores
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"ARI score: {ari_score:.3f}")
    
    # Return negative score since we want to maximize but optimizer minimizes
    return -ari_score

def objective_function_solution(params, solution_features):
    """Optimization objective function for a single solution"""
    eps, min_samples, n_components = params
    min_samples = int(min_samples)  # Convert to integer
    n_components = int(n_components)  # Convert to integer
    
    print(f"\nTrying parameters:")
    print(f"  eps: {eps:.3f}")
    print(f"  min_samples: {min_samples}")
    print(f"  n_components: {n_components}")
    
    # Prepare features with PCA and scaling
    all_features_scaled, pca, scaler = prepare_features(solution_features, n_components)

    # Perform clustering with current parameters
    predicted_labels = perform_clustering(all_features_scaled, eps, min_samples)

    fitness = evaluate_solution_clustering(predicted_labels)
    return fitness

def optimize_single_solution(solution_index):
    """Optimize parameters for a single solution"""
    print(f"\nProcessing solution {solution_index}")
    solution_pieces = get_solution_pieces(solution_index)
    solution_features = get_solution_features(solution_pieces)
    print(f"Number of pieces: {len(solution_pieces)}")
    print(f"Number of features per piece: {len(solution_features[0])}")

    bounds = [
        (0.1, 2.0),     # eps
        (2, 10),        # min_samples 
        (2, len(solution_pieces))  # n_components
    ]

    result = differential_evolution(
        objective_function_solution,
        bounds,
        args=(solution_features,),
        maxiter=1,
        popsize=15,
        disp=True
    )
    
    eps_opt, min_samples_opt, n_components_opt = result.x
    n_components_opt = int(n_components_opt)
    min_samples_opt = int(min_samples_opt)

    print("\nOptimization Results:")
    print("Best parameters found:")
    print(f"  eps: {eps_opt:.3f}")
    print(f"  min_samples: {min_samples_opt}")
    print(f"  n_components: {n_components_opt}")
    print(f"Best fitness score: {-result.fun:.3f}")
    
    return {
        "eps": float(eps_opt),
        "min_samples": min_samples_opt,
        "n_components": n_components_opt,
        "fitness_score": float(-result.fun)
    }

def save_results(results, filename):
    """Save results to a JSON file"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / filename
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    
if __name__ == "__main__":
    all_results = {}
    
    for solution_index in range(3):
        result = optimize_single_solution(solution_index)
        all_results[f"solution_{solution_index}"] = result
    
    save_results(all_results, "clustering_parameters.json")

    quit()

    print("Starting parameter optimization...")
    
    # Load and prepare data
    num_images_to_load = None
    pieces_per_image, labels_per_image = get_pieces_per_image(num_images_to_load)
    features_per_image = extract_features(pieces_per_image)
    true_labels = create_true_labels(features_per_image)

    # Define parameter bounds
    bounds = [
        (0.1, 2.0),     # eps
        (2, 10),         # min_samples 
        (2, 50)         # n_components
    ]

    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        args=(features_per_image, true_labels),
        maxiter=1,
        popsize=15,
        disp=True
    )

    # Get optimized parameters
    eps_opt, min_samples_opt, n_components_opt = result.x
    n_components_opt = int(n_components_opt)

    print("\nOptimization Results:")
    print(f"Best parameters found:")
    print(f"  eps: {eps_opt:.3f}")
    print(f"  min_samples: {int(min_samples_opt)}")
    print(f"  n_components: {n_components_opt}")
    print(f"Best ARI score: {-result.fun:.3f}")

